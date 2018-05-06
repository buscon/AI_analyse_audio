#! /usr/bin/env python3

import pyaudio
import librosa
import numpy as np
import requests

pa = pyaudio.PyAudio()

class RingBuffer():
    "A 1D ring buffer using numpy arrays"
    def __init__(self, length):
        self.data = np.zeros(length, dtype='f')
        self.index = 0

    def extend(self, x):
        "adds array x to ring buffer"
        x_index = (self.index + np.arange(x.size)) % self.data.size
        self.data[x_index] = x
        self.index = x_index[-1] + 1

    def get(self):
        "Returns the first-in-first-out data in the ring buffer"
        idx = (self.index + np.arange(self.data.size)) %self.data.size
        return self.data[idx]


# ring buffer will keep the last 2 seconds worth of audio
ringBuffer = RingBuffer(2 * 44100)

def callback(in_data, frame_count, time_info, flag):
    audio_data = np.fromstring(in_data, dtype=np.float32)
    
    ringBuffer.extend(audio_data)

    # machine learning model takes wavform as input and
    # decides if the last 2 seconds of audio contains a goal
    #if model.is_goal(ringBuffer.get()):
        # GOAL!! Trigger light show
    #    requests.get("http://127.0.0.1:8082/goal")
    print(in_data)
    return (in_data, pa.paContinue)

# function that finds the index of the Soundflower
# input device and HDMI output device
#dev_indexes = findAudioDevices()

stream = pa.open(format = pyaudio.paFloat32,
                 channels = 1,
                 rate = 44100,
                 output = True,
                 input = True,
                 input_device_index = 12,
                 output_device_index = 12,
                 stream_callback = callback)

# start the stream
stream.start_stream()

#while stream.is_active():
#    sleep(0.25)

#stream.close()
#pa.terminate()
