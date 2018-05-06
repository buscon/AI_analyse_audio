#!/usr/bin/env python3
"""Pass input directly to output.

See https://www.assembla.com/spaces/portaudio/subversion/source/HEAD/portaudio/trunk/test/patest_wire.c

"""
import argparse
import logging
import numpy as np

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

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-i', '--input-device', type=int_or_str,
                    help='input device ID or substring')
parser.add_argument('-o', '--output-device', type=int_or_str,
                    help='output device ID or substring')
parser.add_argument('-c', '--channels', type=int, default=2,
                    help='number of channels')
parser.add_argument('-t', '--dtype', help='audio data type')
parser.add_argument('-s', '--samplerate', type=float, help='sampling rate')
parser.add_argument('-b', '--blocksize', type=int, help='block size')
parser.add_argument('-l', '--latency', type=float, help='latency in seconds')
args = parser.parse_args()

try:
    import sounddevice as sd
    import pyaudio as pa

    ringBuffer = RingBuffer(2 * 44100)

    def callback(indata, outdata, frames, time, status):
        if status:
            print(status)
        print(indata)
        ringBuffer.extend(indata)
        outdata[:] = indata
        return(indata, pa.paContinue)


    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=args.samplerate, blocksize=args.blocksize,
                   dtype=args.dtype, latency=args.latency,
                   channels=args.channels, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('\nInterrupted by user')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
