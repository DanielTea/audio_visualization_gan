import numpy as np

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

observations = np.random.rand(1000)
n= 901
print(len(np.std(rolling_window(observations, n), 1)))

# """PyAudio Example: Play a WAVE file."""

# import pyaudio
# import wave
# import sys
# import librosa
# import soundfile as sf
# import numpy as np

# CHUNK = 1024
# maxValue = 2**8
# bars = 100

# filepath = '/Users/dtremer/Documents/GitHub/GANs-Keras-master/Daniel Tremer Track (Master v1).wav'
# print("Plays a wave file.\n\nUsage: %s filename.wav" % filepath)

# x,_ = librosa.load(filepath, sr=48000)
# sf.write('tmp.wav', x, 48000)
# wave.open('tmp.wav','r')

# wf = wave.open('tmp.wav', 'rb')

# p = pyaudio.PyAudio()




# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True)

# data = wf.readframes(CHUNK)

# while True:
#     stream.write(data)
#     #data = wf.readframes(CHUNK)

#     data = np.fromstring(wf.readframes(CHUNK),dtype=np.int8)
#     #dataL = data[0::2]
#     dataR = data[1::2]
#     #peakL = np.abs(np.max(dataL)-np.min(dataL))/maxValue
#     peakR = np.abs(np.max(dataR)-np.min(dataR))/maxValue
#     #lString = "#"*int(peakL*bars)+"-"*int(bars-peakL*bars)
#     rString = "#"*int(peakR*bars)+"-"*int(bars-peakR*bars)
#     print("R=[%s]"%(rString))



# stream.stop_stream()
# stream.close()

# p.terminate()