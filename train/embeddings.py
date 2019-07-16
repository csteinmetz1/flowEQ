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
import matplotlib.pyplot as plt

from models import *
from utils import *

def generate(data, descriptors, arch, weights, samples=3):

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

	codes = OrderedDict({})

	for descriptor_class, descriptor_index in classes.items():
		class_samples = z_mean[np.where(labels == descriptor_index)[0]]
		mean_code = np.mean(class_samples, axis=0)
		stddev_code = np.std(class_samples, axis=0)

		for s in np.arange(samples):
			factor = np.random.choice([-2, -1, 0, 1, 2], replace=False)
			codes[f"{dim}d_{descriptor_class}{s+1}"] = mean_code + (factor * stddev_code)

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
			beat = 3
		elif np.isclose(beta_max, 0.001):
			beta = 2
		elif np.isclose(beta_max, 0.000):
			beta = 1

		c = generate(eq_df, descriptors, a, w)

		for idx, (key, val) in enumerate(c.items()):
			print(idx, key, beta, val)
			code = np.zeros(3)
			code[:val.shape[0]] = val
			#print(code)
			codes[dim-1][beta-1][idx] = code

	sio.savemat('../plugin/assets/codes.mat', {'codes' : codes})	