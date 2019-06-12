from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import pandas as pd

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
train_params = eq_params.reset_index().values[0:1600,1:14]
test_params = eq_params.reset_index().values[1600:,1:14]

# network params
original_dim = 13
input_shape = (original_dim,)
latent_dim = 13
batch_size = 128
epochs = 200

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
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var) # where does this come from?
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5 # why do we do this?

vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

# train !
vae.fit(train_params,
	    epochs=epochs, 
		batch_size=batch_size,
		validation_data=(test_params, None))

	
