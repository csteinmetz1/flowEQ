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

def generate(data, descriptors, arch, weights, visualize=False):

	# model reconstruction from JSON file
	with open(arch, 'r') as f:
		encoder = model_from_json(f.read())

	# load weights into the new model
	encoder.load_weights(weights)

	# generate embeddings for data points
	x = np.array(data.values[:,1:])
	a = encoder.predict(x, batch_size=8)
	z_mean, _, _ = encoder.predict(x, batch_size=8)
	dim = z_mean.shape[1]

	classes = OrderedDict({b: a for a, b in enumerate(set(descriptors))})
	labels = data['descriptor'].map(classes, na_action='ignore').values

	# create linear classifier
	clf = SGDClassifier()
	clf.fit(z_mean, labels)

	codes = OrderedDict({})

	for descriptor_class, descriptor_index in classes.items():
		class_samples = z_mean[np.where(labels == descriptor_index)[0]]

		for factor in [1, 2, 3]:
			codes[f"{dim}d_{descriptor_class}{factor+1}"] = (-(1/clf.coef_) * factor) + clf.intercept_
			
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
			if dim == 3:
				scatter = ax.scatter(code[0], code[1], code[2],
									c=colors[3], label=descriptor_class)
			elif dim == 2:
				scatter = ax.scatter(code[0], code[1], 
									c=colors[3], label=descriptor_class)
			else:
				scatter = ax.scatter(code[0], 0,
									c=colors[3], label=descriptor_class)         

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
	models = glob.glob(os.path.join(args.modeldir, "*.h5"))

	# codes dictionary
	codes = np.empty([3,4,6,3])

	for model in sorted(models):
		w = model
		a = model.replace('.h5', '.json')

		dim  = int(os.path.basename(w)[7])
		beta_max = float(os.path.basename(w)[15:20])
  
		if   np.isclose(beta_max, 0.02):
			beta = 4
		elif np.isclose(beta_max, 0.01):
			beta = 3
		elif np.isclose(beta_max, 0.001):
			beta = 2
		elif np.isclose(beta_max, 0.000):
			beta = 1

		c = generate(eq_df, descriptors, a, w)

		for idx, (key, val) in enumerate(c.items()):
			code = np.zeros(3)
			code[:val.shape[0]] = val
			codes[dim-1][beta-1][idx] = code

	sio.savemat('../plugin/assets/codes.mat', {'codes' : codes})	