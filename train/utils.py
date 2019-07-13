import os
import numpy as np
import pandas as pd
from scipy import signal as sg
import matplotlib.pyplot as plt
from matplotlib import ticker
from nltk.stem import PorterStemmer

ps = PorterStemmer()

# constants for parameter normalization
xmax = np.array([ 12.0, 1000.0,  12.0, 3900.0, 10.0,  12.0, 4700.0, 10.0,  12.0, 10000.0, 10.0,  12.0, 20000.0]);
xmin = np.array([-12.0,   22.0, -12.0,   82.0,  0.1, -12.0,  180.0,  0.1, -12.0,   220.0,  0.1, -12.0,   580.0]);

def normalize_params(x):

    return (x - xmin) / (xmax - xmin);

def denormalize_params(x):
    
    return (x * (xmax - xmin)) + xmin;

def sort_params(x):

    y = x.copy()

    # sort the frequencies 
    sorted_freqs_idx = np.argsort([x[3], x[6], x[9]])

    # check if the order is correct
    if not np.array_equal(sorted_freqs_idx, [0, 1, 2]):

        # fix ordering if first band incorrect
        if   sorted_freqs_idx[0] == 1:
            y[2:5] = x[5:8]
        elif sorted_freqs_idx[0] == 2:
            y[2:5] = x[8:11]
        
        # fix ordering if second band incorrect
        if   sorted_freqs_idx[1] == 0:
            y[5:8] = x[2:5]
        elif sorted_freqs_idx[1] == 2:
            y[5:8] = x[8:11]

        # fix ordering if third band incorrect
        if   sorted_freqs_idx[2] == 1:
            y[8:11] = x[5:8]
        elif sorted_freqs_idx[2] == 0:
            y[8:11] = x[2:5]

    return y

def scale_gains(x):
    pass

def make_lowshelf(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Lowshelf filter.

    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 

    https://www.w3.org/2011/audio/audio-eq-cookbook.html

    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.

    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q 
    aminus1TimesCoso = aminus1 * coso

    # coefs calculation
    b0 = A * (aplus1 - aminus1TimesCoso + beta)
    b1 = A * 2 * (aminus1 - aplus1 * coso)
    b2 = A * (aplus1 - aminus1TimesCoso - beta)
    a0 = aplus1 + aminus1TimesCoso + beta
    a1 = -2 * (aminus1 + aplus1 * coso)
    a2 = aplus1 + aminus1TimesCoso - beta

    # output coefs 
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([a0/a0, a1/a0, a2/a0])

    return b, a

def make_highself(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Highshelf filter.

    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 

    https://www.w3.org/2011/audio/audio-eq-cookbook.html

    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.

    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    aminus1 = A - 1
    aplus1 = A + 1
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    coso = np.cos(omega)
    beta = np.sin(omega) * np.sqrt(A) / Q 
    aminus1TimesCoso = aminus1 * coso

    # coefs calculation
    b0 = A * (aplus1 + aminus1TimesCoso + beta)
    b1 = A * -2 * (aminus1 + aplus1 * coso)
    b2 = A * (aplus1 + aminus1TimesCoso - beta)
    a0 = aplus1 - aminus1TimesCoso + beta
    a1 = 2 * (aminus1 - aplus1 * coso)
    a2 = aplus1 - aminus1TimesCoso - beta

    # output coefs
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([a0/a0, a1/a0, a2/a0])
      
    return b, a

def make_peaking(g, fc, Q, fs=44100):
    """Generate filter coefficients for 2nd order Peaking EQ.

    This function follows the code from the JUCE DSP library 
    which can be found in `juce_IIRFilter.cpp`. 
    
    The design equations are based upon those found in the Cookbook 
    formulae for audio equalizer biquad filter coefficients
    by Robert Bristow-Johnson. 

    https://www.w3.org/2011/audio/audio-eq-cookbook.html

    Args:
        g  (float): Gain factor in dB.
        fc (float): Cutoff frequency in Hz.
        Q  (float): Q factor.
        fs (float): Sampling frequency in Hz.

    Returns:
        tuple: (b, a) filter coefficients 
    """
    # convert gain from dB to linear
    g = np.power(10,(g/20))

    # initial values
    A = np.max([0.0, np.sqrt(g)])
    omega = (2 * np.pi * np.max([fc, 2.0])) / fs
    alpha = np.sin(omega) / (Q * 2)
    c2 = -2 * np.cos(omega)
    alphaTimesA = alpha * A
    alphaOverA = alpha / A

    # coefs calculation
    b0 = 1 + alphaTimesA
    b1 = c2
    b2 = 1 - alphaTimesA
    a0 = 1 + alphaOverA
    a1 = c2
    a2 = 1 - alphaOverA

    # output coefs
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([a0/a0, a1/a0, a2/a0])
    
    return b, a

def params2sos(x, fs):
    """Convert 5 band EQ paramaters to 2nd order sections.

    Takes a vector with shape (13,) of denormalized EQ parameters
    and calculates filter coefficients for each of the 5 filters.
    These coefficients (2nd order sections) are then stored into a
    single (5,6) matrix. This matrix can be fed to scipy.signal.sosfreqz()
    in order to determine the frequency response of the cascase of
    all five biquad filters.

    Args:
        x  (float): Gain factor in dB.       
        fs (float): Sampling frequency in Hz.

    Returns:
        ndarray: filter coefficients for 5 band EQ stored in (5,6) matrix.

        [[b1_0, b1_1, b1_2, a1_0, a1_1, a1_2],  # lowshelf coefficients
         [b2_0, b2_1, b2_2, a2_0, a2_1, a2_2],  # first band coefficients
         [b3_0, b3_1, b3_2, a3_0, a3_1, a3_2],  # second band coefficients
         [b4_0, b4_1, b4_2, a4_0, a4_1, a4_2],  # third band coefficients
         [b5_0, b5_1, b5_2, a5_0, a5_1, a5_2]]  # highshelf coefficients
    """
    # generate filter coefficients from eq params
    b1, a1 = make_lowshelf(x[0],  x[1],  0.71,  fs=fs)
    b2, a2 = make_peaking (x[2],  x[3],  x[4],  fs=fs)
    b3, a3 = make_peaking (x[5],  x[6],  x[7],  fs=fs)
    b4, a4 = make_peaking (x[8],  x[9],  x[10], fs=fs)
    b5, a5 = make_highself(x[11], x[12], 0.71,  fs=fs)

    # stuff coefficents into second order sections structure
    sos = [[np.concatenate([b1, a1])],
           [np.concatenate([b2, a2])],
           [np.concatenate([b3, a3])],
           [np.concatenate([b4, a4])],
           [np.concatenate([b5, a5])]]

    return np.array(sos).reshape(5,6)

def subplot_tf(x, fs, ax, zeroline=True, ticks=False, denorm=True):

    if denorm:
        x = denormalize_params(x)

    # convert eq params to second order sections
    sos = params2sos(x, fs)

    # calcuate filter response
    f, h = sg.sosfreqz(sos, worN=2048, fs=fs)	

    # plot the magnitude respose
    ax.plot(f, 20 * np.log10(abs(h)), 'b')
    ax.set_xscale('log')
    ax.set_xlim([20.0, 20000.0])
    ax.set_ylim([-20, 20])
    if ticks:
        ocmaj = ticker.LogLocator(base=10,numticks=12) 
        ax.xaxis.set_major_locator(locmaj)
    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if zeroline:
        ax.axhline(linewidth=0.5, color='gray', zorder=0)
    ax.tick_params(labelbottom=False, labelleft=False)   
    plt.grid(False, which='both')

def plot_tf(x, fs=44100, plot_title=None, to_file=""):

    if not plot_title:
        plot_title = 'Digital filter frequency response'

    # convert eq params to second order sections
    sos = params2sos(x, fs)

    # calcuate filter response
    f, h = sg.sosfreqz(sos, worN=2048, fs=fs)	

    # plot the magnitude respose
    fig, ax1 = plt.subplots()
    ax1.set_title(plot_title)
    ax1.semilogx(f, 20 * np.log10(abs(h)), 'b')
    ax1.set_ylabel('Amplitude [dB]', color='b')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_xlim([22.0, 20000.0])
    ax1.set_ylim([-20, 20])
    ax1.grid()	# note: make this look prettier
    
    if to_file:
        plt.savefig(to_file)
    else:
        plt.show()
    plt.close()

def mse_tf(a, b, fs=44100):
    
    # convert eq params to second order sections
    sosA = params2sos(denormalize_params(a), fs)
    sosB = params2sos(denormalize_params(b), fs)

    # calcuate filter responses
    fA, hA = sg.sosfreqz(sosA, worN=512, fs=fs)	
    fB, hB = sg.sosfreqz(sosB, worN=512, fs=fs)	

    return np.mean((np.abs(hA) - np.abs(hB))**2)

def compare_tf(a, b, ax=None, fs=44100, to_file=None):

    if not ax:
        fig = plt.figure(figsize=(8,4))
        ax = plt.gca()

    # convert eq params to second order sections
    sosA = params2sos(denormalize_params(a), fs)
    sosB = params2sos(denormalize_params(b), fs)

    # calcuate filter responses
    fA, hA = sg.sosfreqz(sosA, worN=1024, fs=fs)	
    fB, hB = sg.sosfreqz(sosB, worN=1024, fs=fs)	

    mse = np.mean((np.abs(hA) - np.abs(hB))**2)

    # plot the magnitude respose
    plt.title(f"MSE: {mse:0.5f}")
    original, = plt.semilogx(fA, 20 * np.log10(abs(hA)), 'r--')
    reconstructed, = plt.semilogx(fB, 20 * np.log10(abs(hB)), 'b')
    ax.legend(handles=[original, reconstructed], labels=['Original', 'Reconstructed'])
    ax.set_ylabel('Amplitude [dB]', color='b')
    ax.set_xlabel('Frequency [Hz]')
    locmaj = ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)
    ax.set_xlim([22.0, 20000.0])
    ax.set_ylim([-20, 20])
    plt.grid()	# note: make this look prettier
    
    if to_file:
        plt.savefig(to_file)

    return fig

def evaluate_reconstruction(x, x_hat, directory):

    for idx, (a, b) in enumerate(zip(x, x_hat)):
        filename = os.path.join(directory, str(idx))
        compare = compare_tf(a, b, fs=44100, to_file=filename)
        plt.close(compare)

def plot_examples(data, filename):

    plt.figure(figsize=(10,10))
    n = np.ceil(np.sqrt(data.shape[0]))

    for idx, x in enumerate(data):
        ax = plt.subplot(n, n, idx+1)
        subplot_tf(x, 44100, ax)

    plt.tight_layout()
    plt.savefig(filename)

def plot_2d_manifold(models, dim=15, data=None, variational=True, to_file=None):
    """Display a 2D manifold of EQ transfer functions.

    Creates an array of subplots that is dim x dim in size. 
    There are two modes of operation. If no data is passed (default),
    then linearly spaced 2D coordinates are passed to the decoder
    to generate the plot. If data is passed, then these data points
    are passed through the encoder and their latent representations
    are plotted in 2D space. Including labels (categorical encoding) will 
    allow for proper color coding of plotted samples.

    Args:
        models  (tuple) : Encoder and decoder models objects.
        dim     (int)   : Dimensions of the manifold (dim x dim).
        data    (tuple) : Sample and label arrays.
        to_file (str)   : Optional string of filepath for saving figure.
                          Just show the figure otherwise.
    """

    if models:
        # unpack (trained) models
        encoder, decoder = models

        # linearly spaced coordinates corresponding to the 2D plot
        grid_x = np.linspace(-2, 2, dim)
        grid_y = np.linspace(-2, 2, dim)[::-1]

        # create new square figure
        fig1 = plt.figure(figsize=(10, 10))

        # iterate over points in 2D space, plot tf at each point
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                subplot_idx = (i*dim) + j + 1
                z_sample = np.array([[xi, yi]])
                x_decoded = decoder.predict(z_sample)
                x = x_decoded[0].reshape(13,)
                ax = plt.subplot(dim, dim, subplot_idx)
                subplot_tf(x, 44100, ax, denorm=True)

        if to_file:
            fig1.savefig(to_file + "1.png")

        plt.close()

    if data:
        # unpack samples and associated category labels
        samples, labels, classes = data

        if len(classes) > 2:
            colors = ["#003f5c", "#444e86", "#955196", "#dd5182", "#ff6e54", "#ffa600"]
        else:
            colors = ["#444e86", "#ff6e54"]

        if variational:
            z_mean, _, _ = encoder.predict(samples, batch_size=8)
        else:
            z_mean = encoder.predict(samples, batch_size=8)

        fig2, ax = plt.subplots(figsize=(12, 10))

        for descriptor_class, descriptor_index in classes.items():
            class_samples = z_mean[np.where(labels == descriptor_index)[0]]
            scatter = ax.scatter(class_samples[:,0], class_samples[:,1], c=colors[descriptor_index], label=descriptor_class)

        plt.legend()
        #legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        #ax.add_artist(legend1)
        #plt.colorbar()
        plt.xlabel("z[0]")
        plt.ylabel("z[1]")
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
    
        if to_file:
            fig2.savefig(to_file + "2.png")

        plt.close()

def stem(word):
    word = word.lower().strip().split()[0]
    word = word.replace(',', '')
    word = word.replace('brighter', 'bright')
    word = word.replace('airy', 'air')
    return word
