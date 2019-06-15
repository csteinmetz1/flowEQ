from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras import optimizers
from keras.models import Model
from keras.datasets import mnist
from keras import losses
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from utils import *

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    #filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    #z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    #plt.figure(figsize=(12, 10))
    #plt.scatter(z_mean[:, 0], z_mean[:, 1])
    #plt.colorbar()
    #plt.xlabel("z[0]")
    #plt.ylabel("z[1]")
    #plt.savefig(filename)
    #plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of eq curves
    n = 15

    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-10, 10, n)
    grid_y = np.linspace(-10, 10, n)[::-1]

    idx = 1

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            x = x_decoded[0].reshape(13, 1)
            ax = plt.subplot(n, n, idx)
            subplot_tf(x, 44100, ax)
            idx += 1

    plt.show()


def sampling(args):

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0] # what does this do?
    dim = K.int_shape(z_mean)[1] # what does this do?
    # random normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# load normalized data from file
eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)

# split into train and test sets
x_train = eq_params.drop('descriptor', axis=1).reset_index().values[1:1600,1:14]
y_train = eq_params['descriptor'].reset_index().values[0:1,1:14]
x_test = eq_params.drop('descriptor', axis=1).reset_index().values[1600:,1:14]
y_test = eq_params['descriptor'].reset_index().values[1600:,1:14]

# MNIST dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#image_size = x_train.shape[1]
#original_dim = image_size * image_size
#x_train = np.reshape(x_train, [-1, original_dim])
#x_test = np.reshape(x_test, [-1, original_dim])
#x_train = x_train.astype('float32') / 255
#x_test = x_test.astype('float32') / 255

# network params
original_dim = x_train.shape[1]
input_shape = (original_dim,)
latent_dim = 2
batch_size = 512
epochs = 1000

# build encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(1024, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparmaterization trick to allow for backprop
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# make encoder into model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(1024, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# make decoder into model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# stick these together to make vae
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# use mse for reconstruction loss
reconstruction_loss = losses.mean_squared_error(inputs, outputs)
reconstruction_loss *= original_dim # what does this do?

# compute kl divergence for other loss term
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) 
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
optimizer = optimizers.Adam(lr=0.01)
vae.compile(optimizer)
vae.summary()
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

# train !
vae.fit(x_train,
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(x_test, None),
        shuffle=True)

#x = np.array([6.09, 114.77, 3.65, 192.036, 0.23, -12, 915.82, 1.32, -2.13, 444.72, 0.71, -12, 2857.14])
x = x_train[0]
print(x)
#y = normalize_params(x)
#print(y)
y_hat = vae.predict(np.array(x).reshape(1,13))
print(y_hat)
x = denormalize_params(y_hat[0])


models = (encoder, decoder)
data = (x_test, y_test)

plot_results(models,
             data,
             batch_size=batch_size,
             model_name="vae_mlp")
    
