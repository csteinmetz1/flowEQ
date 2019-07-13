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
import matplotlib.pyplot as plt

from models import *
from utils import *

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This script requires TensorFlow 2.0 or above."

# load normalized data from file
eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)
count = pd.read_csv("../data/safe/descriptors.csv", sep=",", index_col=0)

# only use data points within the top 2 occuring descriptors
top_descriptors = count.loc[0:1, 'descriptor'].tolist()
eq_df = eq_params[eq_params['descriptor'].isin(top_descriptors)]

# make train / test split
train_set = 1000
x_train = eq_df.values[:train_set,1:]
y_train = eq_df.values[:train_set,1]
x_test  = eq_df.values[train_set:,1:]
y_test  = eq_df.values[train_set:,1]

# inspect training and testing data
print("Training set   : ", x_train.shape)
print("Traing labels  : ", y_train.shape)
print("Testing set    : ", x_test.shape)
print("Testing labels : ", y_test.shape)

# KL annealing (β) setup
beta = K.variable(0.0)
beta_max = 0.015
klstart = 100
annealtime = 10000

class kl_annealing_callback(tf.keras.callbacks.Callback):
    def __init__(self, beta):
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):
        if epoch > klstart:
            new_beta = min(K.get_value(self.beta) + (1./ annealtime), beta_max)
            K.set_value(self.beta, new_beta)
        print ("Current KL Weight is " + str(K.get_value(self.beta)))

#autoencoder, encoder, decoder = build_single_layer_variational_autoencoder(2, x_train.shape[1])
autoencoder, encoder, decoder = build_single_layer_variational_autoencoder(2, x_train.shape[1], beta)

def make_plots(epoch, logs):
    if (epoch+1) % 100 == 0:
        
        z = encoder.predict(x_test[:1,:])
        x_test_hat = decoder.predict(z[2])
      
        # make directory for current epoch
        #epoch_dir = os.path.join(logdir, f"epoch{epoch+1}")
        #if not os.path.isdir(epoch_dir):
        #    os.makedirs(epoch_dir)

        #compare = evaluate_reconstruction(x_test, x_test_hat, epoch_dir) 
        #buf = io.BytesIO()
        #compare.savefig(buf, format='png')
        #plt.close(compare)

        classes = {b: a for a, b in enumerate(set(top_descriptors))}
        labels = eq_df['descriptor'][:train_set].map(classes, na_action='ignore').values
        models = (encoder, decoder)
        data = (x_train, labels, classes)
        plot_2d_manifold(models, data=data, dim=15, variational=True, to_file=os.path.join(logdir,"plots",f"2d_manifold_{epoch+1}_"))

        #file_writer = tf.summary.create_file_writer(logdir) 
        #with file_writer.as_default():
        #    compare_plot = tf.image.decode_png(buf.getvalue(), channels=0)
        #    compare_plot = tf.expand_dims(compare_plot, 0)
        #    tf.summary.image("Reconstruction", compare_plot, step=epoch+1)

# tensorboard logging setup
logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
plotting_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=make_plots)

# create directories for logging
model_dir = os.path.join(logdir,'models')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

plot_dir = os.path.join(logdir,'plots')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

# train the model
autoencoder.fit(x_train, x_train, 
                shuffle=True,
                validation_data=(x_test,x_test),
                batch_size=8, 
                epochs=1000,
                callbacks=[tensorboard_callback, plotting_callback, kl_annealing_callback(beta)],
                verbose=True)

# save the model weights
autoencoder.save_weights(os.path.join(model_dir,'autoencoder.h5'), save_format='h5')
encoder.save_weights(os.path.join(logdir,'models','encoder.h5'), save_format='h5')
decoder.save_weights(os.path.join(logdir,'models','decoder.h5'), save_format='h5')

# Save the model architectures
with open(os.path.join(model_dir,'autoencoder.json'), 'w') as f:
    f.write(autoencoder.to_json())
with open(os.path.join(model_dir,'encoder.json'), 'w') as f:
    f.write(encoder.to_json())
with open(os.path.join(model_dir,'decoder.json'), 'w') as f:
    f.write(decoder.to_json())