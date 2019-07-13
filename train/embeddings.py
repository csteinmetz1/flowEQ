from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io as sio
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt

from models import *
from utils import *

def generate(data, descriptors, arch, weights):

	# model reconstruction from JSON file
	with open(arch, 'r') as f:
		encoder = model_from_json(f.read())

	# load weights into the new model
	encoder.load_weights(weights)

	# generate embeddings for data points
	x = np.array(data.values[:,1:])
	z_mean, _, _ = encoder.predict(x, batch_size=8)
	
	classes = {b: a for a, b in enumerate(set(descriptors))}
	labels = data['descriptor'].map(classes, na_action='ignore').values

	codes = {}

	for descriptor_class, descriptor_index in classes.items():
		class_samples = z_mean[np.where(labels == descriptor_index)[0]]
		mean_code = np.mean(class_samples, axis=0)
		codes[descriptor_class] = mean_code

	return codes
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate average latent codes for semantic descriptors.')
	parser.add_argument('arch', type=str, help='path to JSON file with the model architecture')
	parser.add_argument('weights',type=str, help='path to h5df file with the model weights')
	parser.add_argument('--output', type=str, help='path to output directory for mat files')
	args = parser.parse_args()

	# load normalized data from file
	eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)
	count = pd.read_csv("../data/safe/descriptors.csv", sep=",", index_col=0)

	# only use data points within the top 20 occuring descriptors
	descriptors = count.loc[0:1, 'descriptor'].tolist()
	eq_df = eq_params[eq_params['descriptor'].isin(descriptors)]

	codes = generate(eq_df, descriptors, args.arch, args.weights)
	sio.savemat('../plugin/assets/codes.mat', {'codes' : codes})

	