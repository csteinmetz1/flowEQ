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

from models import *
from utils import *

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# load normalized data from file
eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)

# only use data points with bright or warm descriptors
eq_df = eq_params.reset_index()#[eq_params['descriptor'].isin(['bright', 'warm'])].reset_index()

# make train / test split
x_train = eq_df.values[:1600,2:]
y_train = eq_df.values[:1600,1]
x_test  = eq_df.values[1600:,2:]
y_test  = eq_df.values[1600:,1]

# inspect training and testing data
print("Training set   : ", x_train.shape)
print("Traing labels  : ", y_train.shape)
print("Testing set    : ", x_test.shape)
print("Testing labels : ", y_test.shape)

autoencoder, encoder, decoder = build_multiple_layer_variational_autoencoder(2, x_train.shape[1])

def make_plots(epoch, logs):
    if epoch % 100 == 0:

        # single example for reconstruction testing
        x = np.array([6.09, 114.77, 3.65, 192.036, 0.23, -12, 915.82, 1.32, -2.13, 444.72, 0.71, -12, 2857.14])
        x = normalize_params(x).reshape(1,13)
        z = encoder.predict(x)
        y = decoder.predict(z[2])
        x_hat = denormalize_params(y[0])

      
        compare = compare_tf(x[0], y[0], to_file=os.path.join(logdir,f"reconstruction_{epoch}")) 
        buf = io.BytesIO()
        compare.savefig(buf, format='png')
        buf.seek(0)

        #d = {'warm': 0, 'bright': 1}
        #labels = eq_df['descriptor'][800:].map(d, na_action='ignore').values
        #models = (encoder, decoder)
        #data = (x_test, labels)
        #plot_2d_manifold(models, dim=15, data=data, to_file=os.path.join(plotdir,'2d_manifold'))

        file_writer = tf.summary.create_file_writer(logdir) 
        with file_writer.as_default():
            image = tf.image.decode_png(buf.getvalue(), channels=0)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            tf.summary.image("Training data", image, step=epoch)

# tensorboard logging setup
logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
plotting_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=make_plots)

# train the model
autoencoder.fit(x_train, x_train, 
                  shuffle=True,
                  validation_data=(x_test,x_test),
                  batch_size=8, 
                  epochs=1500,
                callbacks=[tensorboard_callback, plotting_callback])

autoencoder.save_weights('../models/vae.h5', save_format='h5')
