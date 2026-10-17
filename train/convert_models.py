import os
import glob
import argparse

import h5py
import numpy as np
import scipy.io as sio

# Python version of plugin/ConvertModels.m and plugin/LoadModel.m.
# Extracts the decoder weights from each trained autoencoder and saves them
# as .mat files in plugin/assets, ready to be loaded by plugin/Decoder.m.
# Weights are cast to double as Decoder.m operates on doubles.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert trained Keras models to mat files for the plugin.')
    parser.add_argument('modeldir', type=str, help='path directory containing all model files')
    parser.add_argument('--output', type=str, default=os.path.join('..', 'plugin', 'assets'),
                        help='path to output directory for mat files')
    args = parser.parse_args()

    models = sorted(glob.glob(os.path.join(args.modeldir, 'autoencoders', '*.h5')))
    print(f"Found {len(models)} models in {args.modeldir}")

    os.makedirs(args.output, exist_ok=True)

    for h5path in models:
        matpath = os.path.join(args.output, os.path.basename(h5path).replace('.h5', '.mat'))

        # Decoder.m computes (z * W1) + b1, so kernels keep their Keras (in x out) shape
        # and biases are saved as row vectors (savemat's default for 1-D arrays)
        with h5py.File(h5path, 'r') as f:
            weights = {
                'W1' : np.asarray(f['decoder/dense_3/kernel:0'], dtype=np.float64),
                'b1' : np.asarray(f['decoder/dense_3/bias:0'],   dtype=np.float64),
                'W2' : np.asarray(f['decoder/dense_4/kernel:0'], dtype=np.float64),
                'b2' : np.asarray(f['decoder/dense_4/bias:0'],   dtype=np.float64),
            }

        sio.savemat(matpath, weights)
        print(f"{h5path} -> {matpath}")
