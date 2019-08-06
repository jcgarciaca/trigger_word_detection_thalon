import numpy as np
import time
from pydub import AudioSegment
import random
import sys
import os
import glob
from td_utils import *
import matplotlib.mlab as mlab
from scipy.io.wavfile import write

import wave

from keras.models import load_model

import pyaudio
from Queue import Queue
from threading import Thread


class RealTimeModel():
    def __init__(self):
        #####
        # +/- models 
        # model_path = './../exported_models/V1_full/hola_thalon-04-0.98.h5' with 0.95
        #####

        model_path = './../exported_models/V1_2_full/hola_thalon-18-0.96.h5'
        self.model = load_model(model_path)
        self.model.summary()

        # audio from mic
        self.chunk_duration = 0.5 # Each read length in seconds from mic
        self.fs = 44100 # sampling rate for mic
        self.chunk_samples = int(self.fs * self.chunk_duration) # Each read length in number of samples

        # Each model input data duration in seconds, need to be an integer numbers of chunk_duration
        self.feed_duration = 10
        self.feed_samples = int(self.fs * self.feed_duration)

        assert self.feed_duration/self.chunk_duration == int(self.feed_duration/self.chunk_duration)

        print(self.chunk_duration, self.fs, self.chunk_samples, self.feed_duration, self.feed_samples)

        self.background = AudioSegment.from_wav('./../background/ch01_0.wav') - 20.
        
        self.tmp_filename = './../tmp_audio/instant_audio.wav'

        # Queue to communiate between the audio callback and main thread
        self.q = Queue()

        self.run = True
        self.silence_threshold = 100

        # Run the demo for a timeout seconds
        self.timeout = time.time() + 0.5*60  # 0.5 minutes from now

        # Data buffer for the input wavform
        self.data = np.zeros(self.feed_samples, dtype = 'int16')
        self.frames = []

        self.stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.fs,
            input=True,
            frames_per_buffer=self.chunk_samples,
            # input_device_index=0,
            stream_callback=self.callback)

        self.aux_enable = True

    def insert_audio_clip(self, audio_clip):    
        new_background = self.background[0:len(audio_clip)]
        new_background = new_background.overlay(audio_clip, position = 0)
        return new_background

    def detect_triggerword_spectrum(self, x):
        """
        Function to predict the location of the trigger word.
        
        Argument:
        x -- spectrum of shape (freqs, Tx)
        i.e. (Number of frequencies, The number time steps)

        Returns:
        predictions -- flattened numpy array to shape (number of output time steps)
        """
        # the spectogram outputs  and we want (Tx, freqs) to input into the model
        x  = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)
        predictions = self.model.predict(x)
        return predictions.reshape(-1)

    def has_new_triggerword(self, predictions, threshold=0.8):
        """
        Function to detect new trigger word in the latest chunk of input audio.
        It is looking for the rising edge of the predictions data belongs to the
        last/latest chunk.
        
        Argument:
        predictions -- predicted labels from model
        chunk_duration -- time in second of a chunk
        feed_duration -- time in second of the input to model
        threshold -- threshold for probability above a certain to be considered positive

        Returns:
        True if new trigger word detected in the latest chunk
        """
        predictions = predictions > threshold
        chunk_predictions_samples = int(len(predictions) * self.chunk_duration / self.feed_duration)
        chunk_predictions = predictions[-chunk_predictions_samples:]
        level = chunk_predictions[0]
        for pred in chunk_predictions:
            if pred > level:
                return True
            else:
                level = pred
        return False

    def get_spectrogram(self, data):
        """
        Function to compute a spectrogram.
        
        Argument:
        predictions -- one channel / dual channel audio data as numpy array

        Returns:
        pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
        """
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        return pxx

    def plt_spectrogram(self, data):
        """
        Function to compute and plot a spectrogram.
        
        Argument:
        predictions -- one channel / dual channel audio data as numpy array

        Returns:
        pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
        """
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, _, _, _ = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        return pxx

    def callback(self, in_data, frame_count, time_info, status):
        if time.time() > self.timeout:
            pass
            # self.run = False
        self.frames.append(in_data)        
        if len(self.frames) >= int(self.fs / self.chunk_samples * self.chunk_duration):
            wf = wave.open(self.tmp_filename, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            self.frames = []
            instant_audio = AudioSegment.from_wav(self.tmp_filename)
            instant_audio = self.insert_audio_clip(instant_audio)
            data0 = np.array(instant_audio.get_array_of_samples())
            # data0 = np.frombuffer(in_data, dtype='int16')
            
            if np.abs(data0).mean() < self.silence_threshold:
                print('-')
                return (in_data, pyaudio.paContinue)
            else:
                print('*')
            self.data = np.append(self.data, data0)    
            if len(self.data) > self.feed_samples:
                self.data = self.data[-self.feed_samples:]
                # Process data async by sending a queue.
                self.q.put(self.data)
            return (in_data, pyaudio.paContinue)
        else:
            return (in_data, pyaudio.paContinue)


    def main_run(self):
        while self.run:
            n_data = self.q.get()
            spectrum = self.get_spectrogram(n_data)
            preds = self.detect_triggerword_spectrum(spectrum)
            new_trigger = self.has_new_triggerword(preds)
            if new_trigger:
                print('=====================')

        self.stream.stop_stream()
        self.stream.close()


if __name__ == "__main__":
    print('Creating class')
    obj = RealTimeModel()
    obj.main_run()
