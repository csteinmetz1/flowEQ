import sys
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
from tensorflow.keras import backend as K

def build_simple_autoencoder(latent_dim, input_shape):
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

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                        loss='mean_absolute_error')

    return autoencoder, encoder, decoder

def build_single_layer_autoencoder(latent_dim, input_shape):
    """
    Construct a simple single layer autoencoder.

    This tends to give better performance than bigger AE.
    Also initialization is very important for performance. 
    Trying to restart a few times gives best results.

     1 -> 
     2 -> 
     3 -> 
    ...
    10 -> 

    """

    inputs = tf.keras.Input(shape=(input_shape,))
    x = layers.Dense(1024, activation='relu')(inputs)
    z = layers.Dense(latent_dim, activation='relu')(x)
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(1024, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_shape, activation='sigmoid')(latent_inputs)

    # make encoder and decoder models for use later 
    encoder = tf.keras.Model(inputs, z, name='encoder')
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

    # stick these together to make autoencoder
    outputs = decoder(encoder(inputs))
    autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001),
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

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mean_absolute_error')

    return autoencoder, encoder, decoder

def build_single_layer_variational_autoencoder(latent_dim, input_shape):
    """
    Construct a simple single layer variational autoencoder.

     1 -> 
     2 -> 
     3 -> 
    ...
    10 -> 

    """
    
    beta = 0.001

    # encoder structure (generates mean and log of stddev)
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(units=512, activation='relu')(inputs)
    mu = layers.Dense(latent_dim, activation='linear')(x)
    log_sigma = layers.Dense(latent_dim, activation='linear')(x)

    # sampling layer implements reparameterization trick
    z = layers.Lambda(sample, output_shape=(latent_dim,))([mu, log_sigma])

    # decoder structure (takes latent vector and produces new output)
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(units=512, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_shape, activation='sigmoid')(x)

    # make encoder and decoder models for use later 
    encoder = tf.keras.Model(inputs, [mu, log_sigma, z], name='encoder')
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    encoder.summary()
    decoder.summary()

    # stick these together to make autoencoder
    outputs = decoder(encoder(inputs)[2])
    autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')

    # construct loss function
    def vae_loss(y_true, y_pred):
        recon = losses.mean_absolute_error(inputs, outputs)
        kl_loss = beta * 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1) 
        #kl_loss = K.print_tensor(kl_loss[0])
        return K.mean(recon + kl_loss)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=vae_loss)
    autoencoder.summary()

    return autoencoder

def build_multiple_layer_variational_autoencoder(latent_dim, input_shape):
    """
    Construct a simple single layer variational autoencoder.

     1 -> 
     2 -> 
     3 -> 
    ...
    10 -> 

    """
    
    # encoder structure (generates mean and log of stddev)
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    mu = layers.Dense(latent_dim, activation='linear')(x)
    log_sigma = layers.Dense(latent_dim, activation='linear')(x)

    # sampling layer implements reparameterization trick
    z = layers.Lambda(sample, output_shape=(latent_dim,))([mu, log_sigma])

    # decoder structure (takes latent vector and produces new output)
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(latent_inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(input_shape, activation='sigmoid')(x)

    # make encoder and decoder models for use later 
    encoder = tf.keras.Model(inputs, [mu, log_sigma, z], name='encoder')
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
    encoder.summary()
    decoder.summary()

    # stick these together to make autoencoder
    outputs = decoder(encoder(inputs)[2])
    autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')

    # construct loss function
    def vae_loss(y_true, y_pred):
        recon = losses.mean_absolute_error(inputs, outputs)
        kl_loss = 0.002 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1) 
        return K.mean(recon + kl_loss)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=vae_loss)
    autoencoder.summary()

    return autoencoder, encoder, decoder

def sample(args):
    mu, log_sigma = args
    batch = K.shape(mu)[0] 		# number of samples
    dim = K.int_shape(mu)[1] 	# latent dimension
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
    return mu + K.exp(log_sigma * 0.5) * epsilon

def tune_single_layer_variational_autoencoder(x_train, y_train, x_val , y_val, params):
    """
    Construct a simple single layer variational autoencoder.

     1 -> 
     2 -> 
     3 -> 
    ...
    10 -> 

    """

    latent_dim = 2
    input_shape = 13

    beta = 0.001

    # encoder structure (generates mean and log of stddev)
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(units=params['encoder_units'], activation=params['activation'])(inputs)
    mu = layers.Dense(latent_dim, activation='linear')(x)
    log_sigma = layers.Dense(latent_dim, activation='linear')(x)

    # sampling layer implements reparameterization trick
    z = layers.Lambda(sample, output_shape=(latent_dim,))([mu, log_sigma])

    # decoder structure (takes latent vector and produces new output)
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(units=params['decoder_units'], activation=params['activation'])(latent_inputs)
    outputs = layers.Dense(input_shape, activation='sigmoid')(x)

    # make encoder and decoder models for use later 
    encoder = tf.keras.Model(inputs, [mu, log_sigma, z], name='encoder')
    decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

    # stick these together to make autoencoder
    outputs = decoder(encoder(inputs)[2])
    autoencoder = tf.keras.Model(inputs, outputs, name='autoencoder')

    # construct loss function
    def vae_loss(y_true, y_pred):
        recon = losses.mean_absolute_error(inputs, outputs)
        kl_loss = beta * 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1) 
        #kl_loss = K.print_tensor(kl_loss[0])
        return K.mean(recon + kl_loss)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=vae_loss)

    history = autoencoder.fit(x_train, x_train, 
                    shuffle=True,
                    validation_data=(x_val, x_val),
                    batch_size=8, 
                    epochs=params['epochs'],
                    verbose=False)

    return history, autoencoder