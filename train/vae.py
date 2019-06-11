import numpy as np
import pandas as pd
import tensorflow as tf
tfd = tf.contrib.distributions

# load normalized data from file
eq_params = pd.read_csv("../data/safe/normalized_eq_params.csv", sep=",", index_col=0)

# split into train and test sets
train_params = eq_params.reset_index().values[0:1600,1:14]
test_params = eq_params.reset_index().values[1600:,1:14]

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_params).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_params).shuffle(TEST_BUF).batch(BATCH_SIZE)
