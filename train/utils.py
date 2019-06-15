import numpy as np
import pandas as pd
from scipy import signal as sg
import matplotlib.pyplot as plt
from matplotlib import ticker
from nltk.stem import PorterStemmer

ps = PorterStemmer()

param_max = {
    "gain"  :    12.0,
    "Q"	    :    10.0,
    "freq1" :  1000.0,
    "freq2" :  3900.0,
    "freq3" :  4700.0,
    "freq4" : 10000.0,
    "freq5" : 20000.0
}

param_min = {
    "gain"  :   -12.0,
    "Q"	    :     0.1,
    "freq1" :    22.0,
    "freq2" :    82.0,
    "freq3" :   180.0,
    "freq4" :   220.0,
    "freq5" :   580.0
}

def normalize_param_by_type(x, param_type):
    return (x - param_max[param_type]) / (param_max[param_type] - param_min[param_type])

def normalize_params(x):

    # convert to ndarray
    x = np.array(x)

    # create new array y same size as x
    y = np.zeros(x.shape)

    # low shelf
    y[0]  = (x[0]  - param_min['gain'])  / (param_max['gain'] - param_min['gain'])
    y[1]  = (x[1]  - param_min['freq1']) / (param_max['freq1'] - param_min['freq1'])
    # first band
    y[2]  = (x[2]  - param_min['gain'])  / (param_max['gain'] - param_min['gain'])
    y[3]  = (x[3]  - param_min['freq2']) / (param_max['freq2'] - param_min['freq2'])
    y[4]  = (x[4]  - param_min['Q'])     / (param_max['Q'] - param_min['Q'])
    # second band
    y[5]  = (x[5]  - param_min['gain'])  / (param_max['gain'] - param_min['gain'])
    y[6]  = (x[6]  - param_min['freq3']) / (param_max['freq3'] - param_min['freq3'])
    y[7]  = (x[7]  - param_min['Q'])     / (param_max['Q'] - param_min['Q'])
    # second band
    y[8]  = (x[8]  - param_min['gain'])  / (param_max['gain'] - param_min['gain'])
    y[9]  = (x[9]  - param_min['freq4']) / (param_max['freq4'] - param_min['freq4'])
    y[10] = (x[10] - param_min['Q'])     / (param_max['Q'] - param_min['Q'])
    # high shelf
    y[11] = (x[11] - param_min['gain'])  / (param_max['gain'] - param_min['gain'])
    y[12] = (x[12] - param_min['freq5']) / (param_max['freq5'] - param_min['freq5'])

    return y

def denormalize_params(y):
    
    # convert to ndarray
    y = np.array(y)

    # create new array y same size as x
    x = np.zeros(y.shape)

    # low shelf
    x[0]  = (y[0]  * (param_max['gain']  - param_min['gain'])) + param_min['gain'] 
    x[1]  = (y[1]  * (param_max['freq1'] - param_min['freq1'])) + param_min['freq1']
    # first band
    x[2]  = (y[2]  * (param_max['gain']  - param_min['gain'])) + param_min['gain'] 
    x[3]  = (y[3]  * (param_max['freq2'] - param_min['freq2'])) + param_min['freq2']
    x[4]  = (y[4]  * (param_max['Q']     - param_min['Q'])) + param_min['Q']
    # second band
    x[5]  = (y[5]  * (param_max['gain']  - param_min['gain'])) + param_min['gain'] 
    x[6]  = (y[6]  * (param_max['freq3'] - param_min['freq3'])) + param_min['freq3']
    x[7]  = (y[7]  * (param_max['Q']     - param_min['Q'])) + param_min['Q']
    # second band
    x[8]  = (y[8]  * (param_max['gain']  - param_min['gain'])) + param_min['gain'] 
    x[9]  = (y[9]  * (param_max['freq4'] - param_min['freq4'])) + param_min['freq4']
    x[10] = (y[10] * (param_max['Q']     - param_min['Q'])) + param_min['Q']
    # high shelf
    x[11] = (y[11] * (param_max['gain']  - param_min['gain'])) + param_min['gain'] 
    x[12] = (y[12] * (param_max['freq5'] - param_min['freq5'])) + param_min['freq5']

    return x

def make_lowshelf(g, fc, Q, fs=44100):

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

def subplot_tf(x, fs, ax, zeroline=True, ticks=False, norm=True):

    if norm:
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

def plot_tf(x, fs=44100, to_file=""):

    # convert eq params to second order sections
    sos = params2sos(x, fs)

    # calcuate filter response
    f, h = sg.sosfreqz(sos, worN=2048, fs=fs)	

    # plot the magnitude respose
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
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

def compare_tf(a, b, fs=44100, to_file=""):

    plt.figure(figsize=(8,4))
    ax = plt.gca()

    # convert eq params to second order sections
    sosA = params2sos(denormalize_params(a), fs)
    sosB = params2sos(denormalize_params(b), fs)

    # calcuate filter responses
    fA, hA = sg.sosfreqz(sosA, worN=2048, fs=fs)	
    fB, hB = sg.sosfreqz(sosB, worN=2048, fs=fs)	

    # plot the magnitude respose
    plt.title('Digital filter frequency response')
    original, = plt.semilogx(fA, 20 * np.log10(abs(hA)), 'r--')
    reconstructed, = plt.semilogx(fB, 20 * np.log10(abs(hB)), 'b')
    plt.legend(handles=[original, reconstructed], labels=['Original', 'Reconstructed'])
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')
    locmaj = ticker.LogLocator(base=10,numticks=12) 
    ax.xaxis.set_major_locator(locmaj)
    plt.xlim([22.0, 20000.0])
    plt.ylim([-20, 20])
    plt.grid()	# note: make this look prettier
    
    if to_file:
        plt.savefig(to_file)
    else:
        plt.show()
    plt.close()

def plot_examples(data, filename):

    plt.figure(figsize=(10,10))
    n = np.ceil(np.sqrt(data.shape[0]))

    for idx, x in enumerate(data):
        ax = plt.subplot(n, n, idx+1)
        subplot_tf(x, 44100, ax)

    plt.tight_layout()
    plt.savefig(filename)

def plot_manifold(models, n=15, data=None, batch_size=128):
    """Display a 2D manifold of EQ transfer functions

    # Arguments
        models (tuple): encoder and decoder models
        n (int) : dimensions of the manifold
        data (tuple): test data and label
        batch_size (int): prediction batch size
    """

    encoder, decoder = models
    x_test, y_test = data

    # linearly spaced coordinates corresponding to the 2D plot
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    idx = 1
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            x = x_decoded[0].reshape(13, 1)
            ax = plt.subplot(n, n, idx)
            subplot_tf(x, 44100, ax)
            idx += 1

    plt.show()

def stem(word):
    word = word.lower().strip().split()[0]
    return word
