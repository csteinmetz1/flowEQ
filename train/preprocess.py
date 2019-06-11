import pandas as pd

# read in original dataset and add in proper columns
df = pd.read_csv('../data/safe/SAFEEqualiserUserData.csv', header=None)
df.columns = ["entry", "descriptor", "ip_address", "?", "?",
			  "low_shelf_gain", "low_shelf_freq", 
			  "first_band_gain", "first_band_freq", "first_band_q",
			  "second_band_gain", "second_band_freq", "second_band_q",
			  "third_band_gain", "third_band_freq", "third_band_q",
			  "high_shelf_gain", "high_shelf_freq",
			  "genre", "instrument", "location", 
			  "experience", "age", "nationality", "hash"]

# grab raw parametric eq params from 5 bands
eq_params = df[["low_shelf_gain",   "low_shelf_freq",  
    			"first_band_gain",  "first_band_freq",  "first_band_q", 
    			"second_band_gain", "second_band_freq", "second_band_q", 
    			"third_band_gain",  "third_band_freq",  "third_band_q", 
    			"high_shelf_gain",  "high_shelf_freq"]]  

# extract individual values for normalization
gains = eq_params[["low_shelf_gain", "first_band_gain", "second_band_gain", "third_band_gain", "high_shelf_gain"]]
qs    = eq_params[["first_band_q", "second_band_q", "third_band_q"]]
freq1 = eq_params["low_shelf_freq"]
freq2 = eq_params["first_band_freq"]
freq3 = eq_params[ "second_band_freq"]
freq4 = eq_params["third_band_freq"]
freq5 = eq_params["high_shelf_freq"]

# max and min values for params
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

# apply normalization to shift each between 0 and 1
norm_gains = (gains - min_gain)  / (max_gain - min_gain)
norm_qs    = (qs - min_q)        / (max_q - min_q)
norm_freq1 = (freq1 - min_freq1) / (max_freq1 - min_freq1)
norm_freq2 = (freq2 - min_freq2) / (max_freq2 - min_freq2)
norm_freq3 = (freq3 - min_freq3) / (max_freq3 - min_freq3)
norm_freq4 = (freq4 - min_freq4) / (max_freq4 - min_freq4)
norm_freq5 = (freq5 - min_freq5) / (max_freq5 - min_freq5)

# reconstruct datafram with normalized data
norm_params = pd.concat([norm_gains['low_shelf_gain'],   norm_freq1,
						 norm_gains['first_band_gain'],  norm_freq2, norm_qs['first_band_q'],
						 norm_gains['second_band_gain'], norm_freq3, norm_qs['second_band_q'],
						 norm_gains['third_band_gain'],  norm_freq4, norm_qs['third_band_q'],
						 norm_gains['high_shelf_gain'],  norm_freq5], axis=1)

# save normalized data to file
norm_params.to_csv("../data/safe/normalized_eq_params.csv", sep=",")