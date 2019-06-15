import os
import pandas as pd
from utils import normalize_params, plot_tf, make_highself, make_lowshelf, make_peaking, stem

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
eq_params = df[["descriptor", 
				"low_shelf_gain",   "low_shelf_freq",  
    			"first_band_gain",  "first_band_freq",  "first_band_q", 
    			"second_band_gain", "second_band_freq", "second_band_q", 
    			"third_band_gain",  "third_band_freq",  "third_band_q", 
    			"high_shelf_gain",  "high_shelf_freq"]]  

# plot transfer function for each set of eq params and save
if not os.path.isdir("../data/safe/plots"):
	os.makedirs("../data/safe/plots")
for index, row in eq_params.iterrows():
	d = row['descriptor']
	filename = f"../data/safe/plots/{index}_{d}.png"
	#print(index, d)
	#plot_tf(row.drop("descriptor"), to_file=filename)

# find most common descriptors
eq_params['descriptor'] = eq_params['descriptor'].map(stem)
print(eq_params.groupby('descriptor').count().sort_values(by=['low_shelf_gain'], ascending=False))

# normalize eq parameters with utility func (then add back in descriptors)
norm_params = eq_params.drop("descriptor", axis=1).apply(normalize_params, axis=1)
norm_params.insert(0, "descriptor", eq_params["descriptor"])

# save normalized data to file
norm_params.to_csv("../data/safe/normalized_eq_params.csv", sep=",")

