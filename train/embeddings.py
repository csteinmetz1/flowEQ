from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
import glob
import argparse
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt

from models import *
from utils import *

def generate(data, descriptors, dim, beta, encoder, decoder, classifier=True, visualize=False):
    """Generate latent space embeddings (or codes) for each model and given semantic descriptors.

    This function takes in the training data (or testing) along with a set of encoder and decoder models.
    We then use the Stochastic Gradient Descent algorithm in sci-kit learn in order to separate the two descriptors
    within the latent space. Using this line we draw a perpendicular line and move along that line in either direction,
    where each direction moves towards a code that is more heavily linked to the respective semantic descriptor. 

    Our current method here for determining these codes is fairly clunky and ideal. 
    It does sometimes produce codes that are not extremely representative of their class, 
    but overall it works well enough for the time being.

    Args:
        data        (DataFrame) : Pandas dataframe with equalizer parameter data (normalized)
        descriptors (list)   	: List of the semantic descriptors as strings
        dim    		(int)   	: Latent dimensions of the autoencoder model
        beta    	(float) 	: Disentanglement factor for the supplied models
        encoder 	(model)   	: Trained encoder Keras model
        decoder		(model)		: trained decoder Keras model
        classifier	(bool)		: Use a linear classifier, otherwise randomally sample data
        visualize	(bool)		: Create scatter plots of data and generated embeddings
    """
    # generate embeddings for data points
    x = np.array(data.values[:,1:])
    a = encoder.predict(x, batch_size=8)
    z_mean, _, _ = encoder.predict(x, batch_size=8)

    classes = OrderedDict({b: a for a, b in enumerate(set(descriptors))})
    labels = data['descriptor'].map(classes, na_action='ignore').values

    codes = OrderedDict({})

    for descriptor_class, descriptor_index in classes.items():
        class_samples = z_mean[np.where(labels == descriptor_index)[0]]

        if classifier:
            # create linear classifier
            clf = SGDClassifier()
            clf.fit(z_mean, labels)

            for factor in [0.5, 1, 2]:
                code = -(clf.intercept_[0]/clf.coef_[0]) + factor
                if clf.predict([code]) != descriptor_index:
                    code = -(clf.intercept_[0]/clf.coef_[0]) - factor
                codes[f"{dim}d_{descriptor_class}{factor+1}"] = code

                x = denormalize_params(decoder.predict(np.array([code])))[0]
                plot_filename = os.path.join("plots", "embeddings", f"{code}.png")
                #print(plot_filename)
                plot_tf(x, plot_title=f"{dim}d_{descriptor_class}{factor+1}", to_file=plot_filename)
        else:
            for factor in np.arange(0,3):
                code_idx = np.random.choice(class_samples.shape[0])
                code = class_samples[code_idx,:]
                codes[f"{dim}d_{descriptor_class}{factor+1}"] = code
                print(code)

    if visualize:

        colors = ["#444e86", "#ff6e54", "#dd5182", "#955196"]

        if dim == 1:
            fig, ax = plt.subplots(figsize=(12, 10))			
        elif dim == 2:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

        for descriptor_class, descriptor_index in classes.items():
            class_samples = z_mean[np.where(labels == descriptor_index)[0]]
            if dim == 3:
                scatter = ax.scatter(class_samples[:,0], class_samples[:,1], class_samples[:,2],
                                    c=colors[descriptor_index], label=descriptor_class)
            elif dim == 2:
                scatter = ax.scatter(class_samples[:,0], class_samples[:,1], 
                                    c=colors[descriptor_index], label=descriptor_class)
            else:
                scatter = ax.scatter(class_samples[:,0], (np.ones(class_samples[:,0].shape) * descriptor_index)/4, 
                                    c=colors[descriptor_index], label=descriptor_class)         
        
        for idx, (descriptor_class, code) in enumerate(codes.items()):

            if idx < 3:
                c_offset = 0
            else:
                c_offset = 1

            if dim == 3:
                scatter = ax.scatter(code[0], code[1], code[2],
                                    c=colors[2+c_offset], label=descriptor_class)
            elif dim == 2:
                scatter = ax.scatter(code[0], code[1], 
                                    c=colors[2+c_offset], label=descriptor_class)
            else:
                scatter = ax.scatter(code[0], 0,
                                    c=colors[2+c_offset], label=descriptor_class)         

        plt.show()       

    return codes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate average latent codes for semantic descriptors.')
    parser.add_argument('modeldir', type=str, help='path directory containing all model files')
    parser.add_argument('--output', type=str, help='path to output directory for mat file')
    args = parser.parse_args()

    # load normalized data from file
    eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)
    count = pd.read_csv("../data/safe/descriptors.csv", sep=",", index_col=0)

    # only use data points within the top 2 occuring descriptors
    descriptors = count.loc[0:1, 'descriptor'].tolist()
    eq_df = eq_params[eq_params['descriptor'].isin(descriptors)]

    # get models
    encoder_models = glob.glob(os.path.join(args.modeldir, 'encoders', "*.h5"))
    decoder_models = glob.glob(os.path.join(args.modeldir, 'decoders', "*.h5"))

    # codes dictionary
    codes = np.empty([3,4,6,3])

    for encoder_model, decoder_model in zip(sorted(encoder_models), sorted(decoder_models)):

        encoder_w = encoder_model
        encoder_a = encoder_model.replace('.h5', '.json')

        # model reconstruction from JSON file
        with open(encoder_a, 'r') as f:
            encoder = model_from_json(f.read())

        # load weights into the new model
        encoder.load_weights(encoder_w)

        decoder_w = decoder_model
        decoder_a = decoder_model.replace('.h5', '.json')

        # model reconstruction from JSON file
        with open(decoder_a, 'r') as f:
            decoder = model_from_json(f.read())

        # load weights into the new model
        decoder.load_weights(decoder_w)

        dim  = int(os.path.basename(encoder_w)[7])
        beta_max = float(os.path.basename(encoder_w)[15:20])
  
        if   np.isclose(beta_max, 0.02):
            beta = 4
        elif np.isclose(beta_max, 0.01):
            beta = 3
        elif np.isclose(beta_max, 0.001):
            beta = 2
        elif np.isclose(beta_max, 0.000):
            beta = 1

        c = generate(eq_df, descriptors, dim, beta, encoder, decoder)

        for idx, (key, val) in enumerate(c.items()):
            code = np.zeros(3)
            code[:val.shape[0]] = val
            print(dim, beta, idx, code)
            codes[dim-1][beta-1][idx] = code

    # check if directory exists
    if not os.path.isdir(os.path.join('..','plugin','assets')):
        os.makedirs(os.path.join('..','plugin','assets'))

    # save the final array into a mat file in the plugin assets directory
    sio.savemat(os.path.join('..','plugin','assets', 'codes.mat'), {'codes' : codes})	