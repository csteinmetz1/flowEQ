import tensorflow as tf
from tensorflow.keras import layers

def build_single_layer_autoencoder(latent_dim, input_shape):
	"""
	Construct a simple single layer autoencoder.

	This tends to give better performance than bigger AE.
	Also initialization is very important for performance. 
	Trying to restart a few times gives best results.

	 1 -> 0.0914
	 2 -> 0.0777
	 3 -> 0.0656
	...
	10 -> 0.0208

	"""

	inputs = tf.keras.Input(shape=(input_shape,))
	z = layers.Dense(latent_dim, activation='relu')(inputs)
	latent_inputs = tf.keras.Input(shape=(latent_dim,))
	outputs = layers.Dense(input_shape, activation='sigmoid')(latent_inputs)

	# make encoder and decoder models for use later 
	encoder = tf.keras.Model(inputs, z, name='encoder')
	decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

	# stick these together to make autoencoder
	outputs = decoder(encoder(inputs))
	autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')

	autoencoder.compile(optimizer=tf.train.AdamOptimizer(0.001),
				        loss='mean_absolute_error')

	return autoencoder, encoder, decoder

def build_multiple_layer_autoencoder(latent_dim, input_shape):	
	"""
	Construct an autoencoder with more layers.

	"""

	inputs = tf.keras.Input(shape=(input_shape,))
	x = layers.Dense(13, activation='relu')(inputs)
	x = layers.Dense(9, activation='relu')(x)
	x = layers.Dense(6, activation='relu')(x)
	x = layers.Dense(2, activation='relu')(x)
	z = layers.Dense(latent_dim, activation='relu')(x)
	latent_inputs = tf.keras.Input(shape=(latent_dim,))
	x = layers.Dense(2, activation='relu')(latent_inputs)
	x = layers.Dense(6, activation='relu')(x)
	x = layers.Dense(9, activation='relu')(x)
	x = layers.Dense(13, activation='relu')(x)
	outputs = layers.Dense(input_shape, activation='sigmoid')(x)

	# make encoder and decoder models for use later 
	encoder = tf.keras.Model(inputs, z, name='encoder')
	decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

	# stick these together to make autoencoder
	outputs = decoder(encoder(inputs))
	autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
				  loss='mean_absolute_error')

	return autoencoder, encoder, decoder