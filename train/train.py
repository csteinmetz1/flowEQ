from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from datetime import datetime
from packaging import version

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from models import *
from utils import *

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

autoencoder, encoder, decoder = build_multiple_layer_variational_autoencoder(2, x_train.shape[1])

# tensorboard logging setup
logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# train the model
autoencoder.fit(x_train, x_train, 
		  		shuffle=True,
		  		validation_data=(x_test,x_test),
		  		batch_size=8, 
		  		epochs=1500,
				callbacks=[tensorboard_callback])

autoencoder.save_weights('../models/vae.h5', save_format='h5')

x = np.array([6.09, 114.77, 3.65, 192.036, 0.23, -12, 915.82, 1.32, -2.13, 444.72, 0.71, -12, 2857.14])
x = normalize_params(x).reshape(1,13)
z = encoder.predict(x)
y = decoder.predict(z[2])
x_hat = denormalize_params(y[0])

print(x[0])
print(y[0])

d = {'warm': 0, 'bright': 1}
labels = eq_df['descriptor'][800:].map(d, na_action='ignore').values
compare_tf(x[0], y[0], to_file=os.path.join(logdir,'reconstruction'))
models = (encoder, decoder)
data = (x_test, labels)
plot_2d_manifold(models, dim=15, data=data, to_file=os.path.join(logdir,'2d_manifold'))