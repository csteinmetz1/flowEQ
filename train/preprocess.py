import os
import pandas as pd
from utils import normalize_params, plot_tf, make_highself, make_lowshelf, make_peaking

# read in original dataset 
df = pd.read_csv('../data/safe/SAFEEqualiserUserData.csv', header=None)

# and add in proper columns
df.columns = ["entry", "descriptor", "ip_address", "?", "?",
			  "low_shelf_gain", "low_shelf_freq", 
			  "first_band_gain", "first_band_freq", "first_band_q",
			  "second_band_gain", "second_band_freq", "second_band_q",
			  "third_band_gain", "third_band_freq", "third_band_q",
			  "high_shelf_gain", "high_shelf_freq",
			  "genre", "instrument", "location", 
			  "experience", "age", "nationality", "hash"]

# grab raw parametric eq params from 5 bands and descriptors
eq_params = df[["low_shelf_gain",   "low_shelf_freq",  
    			"first_band_gain",  "first_band_freq",  "first_band_q", 
    			"second_band_gain", "second_band_freq", "second_band_q", 
    			"third_band_gain",  "third_band_freq",  "third_band_q", 
    			"high_shelf_gain",  "high_shelf_freq"]]  

# plot transfer function for each set of eq params and save
if not os.path.isdir("../data/safe/plots"):
	os.makedirs("../data/safe/plots")
for index, row in eq_params.iterrows():
	print(index)
	plot_tf(row, to_file=f"../data/safe/plots/{index}.png")

# normalize eq parameters with utility func
norm_params = eq_params.apply(normalize_params, axis=1)
# note: this is by row but may be able to make it by column for speed

# save normalized data to file
norm_params.to_csv("../data/safe/normalized_eq_params.csv", sep=",")

