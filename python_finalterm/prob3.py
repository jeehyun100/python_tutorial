from scipy.io.wavfile import read
import matplotlib.pyplot as plt

#import pyaudio
#import winsound

#import pygame
import wave

# read audio samples
source_rate, source_sig = read("q2.wav")
audio = source_sig
# plot the first 1024 samples
plt.plot(audio)
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title
plt.title("Sample Wav")
# display the plot
plt.show()




duration_seconds = len(source_sig) / float(source_rate)
print(duration_seconds)

"""
import numpy as np
from scipy.io import wavfile
from scipy import interpolate

NEW_SAMPLERATE = 48000

old_samplerate, old_audio = wavfile.read("test.wav")

if old_samplerate != NEW_SAMPLERATE:
    duration = old_audio.shape[0] / old_samplerate

    time_old  = np.linspace(0, duration, old_audio.shape[0])
    time_new  = np.linspace(0, duration, int(old_audio.shape[0] * NEW_SAMPLERATE / old_samplerate))

    interpolator = interpolate.interp1d(time_old, old_audio.T)
    new_audio = interpolator(time_new).T

    wavfile.write("out.wav", NEW_SAMPLERATE, np.round(new_audio).astype(old_audio.dtype))

"""

"""
#!usr/bin/env python  
#coding=utf-8  

import pyaudio  
import wave  

#define stream chunk   
chunk = 1024  

#open a wav format music  
f = wave.open(r"/usr/share/sounds/alsa/Rear_Center.wav","rb")  
#instantiate PyAudio  
p = pyaudio.PyAudio()  
#open stream  
stream = p.open(format = p.get_format_from_width(f.getsampwidth()),  
                channels = f.getnchannels(),  
                rate = f.getframerate(),  
                output = True)  
#read data  
data = f.readframes(chunk)  

#play stream  
while data:  
    stream.write(data)  
    data = f.readframes(chunk)  

#stop stream  
stream.stop_stream()  
stream.close()  

#close PyAudio  
p.terminate() 
"""