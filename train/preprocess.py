import os
import pandas as pd
from utils import *

# read in original dataset 
df_user  = pd.read_csv('../data/safe/SAFEEqualiserUserData.csv', header=None)
#df_audio = pd.read_csv('../data/safe/SAFEEqualiserAudioFeatureData.csv', header=None)


# and add in proper columns
df_user.columns = ["entry", "descriptor", "ip_address", "?", "?",
                   "low_shelf_gain", "low_shelf_freq", 
                   "first_band_gain", "first_band_freq", "first_band_q",
                   "second_band_gain", "second_band_freq", "second_band_q",
                   "third_band_gain", "third_band_freq", "third_band_q",
                   "high_shelf_gain", "high_shelf_freq",
                   "genre", "instrument", "location", 
                   "experience", "age", "nationality", "hash"]

#df_audio.columns = ["entry", "type", "mean", "variance", "stddev", "?",
#                    "rms_amplitude", "zero_crossing_rate", 
#                    "spectral_centroid", "spectral_variance", "spectral_stddev",
#                    "spectral_skewness", "spectral_kurtosis", 
#                    "irregularity_j", "irregularity_k",
#                    "f0", "smoothness", "spectral_roll_off", "spectral_flatness",
#                    "spectral_crest", "spectral_slope", "peak_spectral_centroid",
#                    "peak_spectral_variance", "peak_spectral_stddev", "peak_spectral_skewness",
#                    "peak_spectral_kurtosis", "peak_irregularity_j", "peak_irregularity_k",
#                    "peak_tristimulis1", "peak_tristimulis2", "peak_tristimulis3",
#                    "inharmonicity", 
#                    "harmonic_spectral_centroid", "harmonic_spectral_variance",
#                    "harmonic_spectral_stddev", "harmonic_spectral_skewness",
#                    "harmonic_spectral_kurtosis", 
#                    "harmonic_irregularity_j", "harmonic_irregularity_k",
#                    "harmonic_tristimulis1", "harmonic_tristimulis2", "harmonic_tristimulis3",
#                    "noisiness", "parity_ratio",
#                    "bark_coef1", "bark_coef2", "bark_coef3", "bark_coef4", "bark_coef5",
#                    "bark_coef6", "bark_coef7", "bark_coef8", "bark_coef9", "bark_coef10", 
#                    "bark_coef11", "bark_coef12", "bark_coef13", "bark_coef14", "bark_coef15",
#                    "bark_coef16", "bark_coef17", "bark_coef18", "bark_coef19", "bark_coef20",
#                    "bark_coef21", "bark_coef22", "bark_coef23", "bark_coef24", "bark_coef25",
#                    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7", "mfcc8", 
#                    "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13"]

#pd.set_option('display.max_columns', None)
#print(df_audio.head())

# grab raw parametric eq params from 5 bands and descriptors
eq_params = df_user[["descriptor", 
                     "low_shelf_gain",   "low_shelf_freq",  
                     "first_band_gain",  "first_band_freq",  "first_band_q", 
                     "second_band_gain", "second_band_freq", "second_band_q", 
                     "third_band_gain",  "third_band_freq",  "third_band_q", 
                     "high_shelf_gain",  "high_shelf_freq"]]  

# plot transfer function for each set of eq params and save
if not os.path.isdir("../data/safe/plots"):
    os.makedirs("../data/safe/plots")

print("Generating transfer function plots from dataset...")
for index, row in eq_params.iterrows():
    d = row['descriptor']
    filename = f"../data/safe/plots/{index}_{d}.png"
    plot_tf(row.drop("descriptor"), plot_title=d, to_file=filename)

# find most common descriptors 
eq_params['descriptor'] = eq_params['descriptor'].map(stem) # this throws a warning - it's fine but should fix later
count = eq_params.groupby('descriptor').count().sort_values(by=['low_shelf_gain'], ascending=False)[["low_shelf_gain"]]
count = count.rename(index=str, columns={'low_shelf_gain' : 'count'})
count = count.reset_index()
count.to_csv("../data/safe/descriptors.csv", sep=",")

# sort three parametric bands by center frequency
sorted_params = eq_params.drop("descriptor", axis=1).apply(sort_params, axis=1)

# normalize eq parameters with utility func (then add back in descriptors)
norm_params = sorted_params.apply(normalize_params, axis=1)

# put the descriptors back in
norm_params.insert(0, "descriptor", eq_params["descriptor"])

# shuffle the rows before training
norm_params = norm_params.sample(frac=1).reset_index(drop=True)

# save normalized and shuffled data to file
norm_params.to_csv("../data/safe/normalized_eq_params.csv", sep=",")