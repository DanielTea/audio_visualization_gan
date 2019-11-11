import matplotlib.pyplot as plt
from utils.utils import z_noise, c_noise
from keras.utils.np_utils import to_categorical
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import pylab as pl

import pyaudio
import wave
import sys
import librosa
import soundfile as sf
import numpy as np

import cv2
from scipy.signal import savgol_filter

from scipy.signal import lfilter
from scipy.signal import butter

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)





def plot_results_GAN(G, n=4):

    def results(G,n, noize):

        img = np.zeros((n * 28,1))


        col = np.multiply(np.add(G.predict(noize).reshape(n*28, 28),1.0), 255.0/2.0)

       # col = np.add(G.predict(noize).reshape(n*28, 28),1.0)

        img = np.concatenate((img,col), axis=1)


        return img

    """ Plots n x n windows from DCGAN and WGAN generator
    """

    """PyAudio Example: Play a WAVE file."""

    CHUNK = 1024
    maxValue = 2**8
    bars = 100

    filepath = '/Users/dtremer/Documents/GitHub/GANs-Keras-master/Daniel Tremer Track (Master v1).wav'
    print("Plays a wave file.\n\nUsage: %s filename.wav" % filepath)

    x,_ = librosa.load(filepath, sr=48000)
    sf.write('tmp.wav', x, 48000)
    wave.open('tmp.wav','r')

    wf = wave.open('tmp.wav', 'rb')

    p = pyaudio.PyAudio()


    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)
    fr = wf.getframerate()
    print(wf.getframerate())

    timestamp = 0

    while True:
        timestamp +=1
        
        stream.write(data)
        #data = wf.readframes(CHUNK)

        

        data = np.fromstring(wf.readframes(CHUNK),dtype=np.int8)
        dataL = data[0::2]
        dataR = data[1::2]
        print(dataR.shape)
        print(dataR[:100])

        observations_R = dataR[0:250]
        observations_L = dataR[250:500]
        observations_R_1 = dataR[500:750]
        observations_L_2 = dataL[750:1000]
        

        n_= 151
        observations_r= np.std(rolling_window(observations_R, n_), 1)
        observations_l= np.std(rolling_window(observations_L, n_), 1)
        observations_r_1= np.std(rolling_window(observations_R_1, n_), 1)
        observations_l_2= np.std(rolling_window(observations_L_2, n_), 1)
        

        seed_mat = np.asarray([
            np.asarray(observations_r),
            np.asarray(observations_l),
            np.asarray(observations_r_1),
            np.asarray(observations_l_2)

        ])

        # seed_mat= [
        #     dataR[:100],
        #     dataR[100:200]
        # ]

        print(seed_mat)

        # scaler = MinMaxScaler()
        # scaler.fit(seed_mat)


        # #peakL = np.abs(np.max(dataL)-np.min(dataL))/maxValue
        # peakR = np.abs(np.max(dataR)-np.min(dataR))/maxValue
        # #lString = "#"*int(peakL*bars)+"-"*int(bars-peakL*bars)
        # rString = "#"*int(peakR*bars)+"-"*int(bars-peakR*bars)
        # print("R=[%s]"%(rString))

        if timestamp % 3 == 0:

            # scaler = MinMaxScaler()
            # scaler.fit(seed_mat)

            img = results(G,n=n, noize=seed_mat)
            

            scale_percent = 500 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            

            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            

            cv2.imshow('frame',resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()

def plot_large(img):
    """ Custom sized image
    """
    fig1 = plt.figure(figsize = (4,4))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.imshow(img, cmap='gray')
    plt.show()

def plot_results_CGAN(G):
    """ Plots n x n windows from CGAN generator
    """
    labels = np.arange(0, 10)

    n = len(labels)
    img = np.zeros((n * 28,1))
    for i in range(n):
        # Remap from tanh range [-1, 1] to image range [0, 255]
        col = np.multiply(np.add(G.predict([z_noise(n), \
            to_categorical(labels,n)]).reshape(n * 28,28), 1.0), 255.0/2.0)
        img = np.concatenate((img,col), axis=1)
    plot_large(img)

def plot_results_InfoGAN(G):
    """ Plots 10x10 windows from InfoGAN generator
    """
    # Latent code
    lower = 0.2
    upper = 0.8
    latent_code =  np.arange(lower, upper, (upper-lower)/10.0)

    # Fixed input
    n_rows = 10
    z = np.ones((n_rows, 100))
    img = np.zeros((n_rows*28,1))
    
    for i in range(10):
        x = np.zeros((n_rows,10))
        x[:,i] = latent_code
        # Convert tanh range [-1; 1] to [0; 255]
        col = np.multiply(np.add(G.predict([z,x]).reshape(n_rows*28,28), 1.0), 255.0/2.0)
        img = np.concatenate((col,img), axis=1)
    plot_large(img)
