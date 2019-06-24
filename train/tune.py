from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
from datetime import datetime
from packaging import version

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import talos as ta

from models import *
from utils import *

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# load normalized data from file
eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)

# only use data points with bright or warm descriptors
eq_df = eq_params[eq_params['descriptor'].isin(['bright', 'warm'])].reset_index()

# make train / test split
x_train = eq_df.values[:800,2:]
y_train = eq_df.values[:800,1]
x_test  = eq_df.values[800:,2:]
y_test  = eq_df.values[800:,1]

# inspect training and testing data
print("Training set   : ", x_train.shape)
print("Traing labels  : ", y_train.shape)
print("Testing set    : ", x_test.shape)
print("Testing labels : ", y_test.shape)

# setup hyperparameter search space
p = {'activation':['relu'],
     'encoder_units': np.arange(16, 1024, step=32),
     'decoder_units': np.arange(16, 1024, step=32),
     'epochs': np.arange(10, 500, step=100)}

# perform the search
ta.Scan(x=x_train, y=x_train, x_val=x_test, y_val=x_test,
        model=tune_single_layer_variational_autoencoder, 
        params=p, grid_downsample=0.1, clear_tf_session=False)