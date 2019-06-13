from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

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

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def denormalize_eq_params(x):

    min_gain  =   -12.00
    max_gain  =    12.00
    min_q     =     0.71
    max_q     =    10.00
    min_freq1 =   150.00
    max_freq1 =  1000.00
    min_freq2 =   560.00
    max_freq2 =  3900.00
    min_freq3 =  1000.00
    max_freq3 =  4700.00
    min_freq4 =  3300.00
    max_freq4 = 10000.00
    min_freq5 =  8200.00
    max_freq5 = 20000.00

    n_gain  = (max_gain - min_gain) + min_gain
    n_q     = (max_q -  min_q) + min_q
    n_freq1 = (max_freq1 - min_freq1) + min_freq1
    n_freq2 = (max_freq2 - min_freq2) + min_freq2
    n_freq3 = (max_freq3 - min_freq3) + min_freq3
    n_freq4 = (max_freq4 - min_freq4) + min_freq4
    n_freq5 = (max_freq5 - min_freq5) - min_freq5

    y = x

    y[0] = x[0] * n_gain
    y[1] = x[1] * n_freq1
    y[2] = x[2] * n_gain
    y[3] = x[3] * n_freq2
    y[4] = x[4] * n_q
    y[5] = x[5] * n_gain
    y[6] = x[6] * n_freq3
    y[7] = x[7] * n_q	
    y[8] = x[8] * n_gain
    y[9] = x[9] * n_freq4
    y[10] = x[10] * n_q	
    y[11] = x[11] * n_gain
    y[12] = x[2] * n_freq5

    return y
 
def print_eq_params(x):
    print(f"lowshelf gain: {x[0]:0.2f}")
    print(f"lowshelf freq: {x[1]:0.2f}")

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
x_train = eq_params.reset_index().values[0:1600,1:14]
x_test = eq_params.reset_index().values[1600:,1:14]


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
batch_size = 128
epochs = 10

# build encoder
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(64, activation='relu')(inputs)
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
x = Dense(64, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# make decoder into model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# stick these together to make vae
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

# use mse for reconstruction loss
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim # what does this do?

# compute kl divergence for other loss term
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) 
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

# train !
vae.fit(x_train,
        epochs=epochs, 
        batch_size=batch_size,
        validation_data=(x_test, None))

for eq_params in x_test:
    x = np.array(eq_params)
    print_eq_params(x)
    #z = encoder.predict(x.reshape(1,13,))
    #x_hat = decoder.predict(z[2])

#models = (encoder, decoder)
#data = (x_test, y_test)

#plot_results(models,
#             data,
#             batch_size=batch_size,
#             model_name="vae_mlp")
    
