import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from models import *
from utils import *

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

autoencoder, encoder, decoder = build_single_layer_autoencoder(10, x_train.shape[1])

# train the model
autoencoder.fit(x_train, x_train, 
		  		shuffle=True,
		  		validation_data=(x_test,x_test),
		  		batch_size=8, 
		  		epochs=100)


z = np.array(np.zeros([1, 10]))
y = decoder.predict(z)
print(y[0])
x = denormalize_params(y[0])
print(x)
plot_tf(x)