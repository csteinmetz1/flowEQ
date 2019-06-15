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

plot_examples(x_test[:225], "plots/random_examples.png")