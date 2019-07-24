import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./raw_data/activates"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./raw_data/activates/"+filename)
            activates.append(activate)
    for filename in os.listdir("./raw_data/backgrounds"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./raw_data/backgrounds/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./raw_data/negatives"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./raw_data/negatives/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds
    
# Load raw audio files for speech synthesis
def load_raw_audio_thalon():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./thalon_audios/filtered"):
        if filename.endswith("wav"):
            activate = AudioSegment.from_wav("./thalon_audios/filtered/"+filename)
            activates.append(activate)
    for filename in os.listdir("./database/wav_resampling/backgrounds_10sec"):
        if filename.endswith("wav"):
            background = AudioSegment.from_wav("./database/wav_resampling/backgrounds_10sec/"+filename)
            backgrounds.append(background)
    for filename in os.listdir("./database/wav_resampling/general_lt_1sec"):
        if filename.endswith("wav"):
            negative = AudioSegment.from_wav("./database/wav_resampling/general_lt_1sec/"+filename)
            negatives.append(negative)
    return activates, negatives, backgrounds

def load_raw_audio_thalon_list():
    activates = []
    backgrounds = []
    negatives = []
    for filename in os.listdir("./thalon_audios/filtered"):
        if filename.endswith("wav"):
            activates.append("./thalon_audios/filtered/"+filename)
    for filename in os.listdir("./database/wav_resampling/backgrounds_10sec"):
        if filename.endswith("wav"):
            backgrounds.append("./database/wav_resampling/backgrounds_10sec/"+filename)
    for filename in os.listdir("./database/wav_resampling/general_lt_1sec"):
        if filename.endswith("wav"):
            negatives.append("./database/wav_resampling/general_lt_1sec/"+filename)
    return activates, negatives, backgrounds

def load_raw_data(audio_names):
    audio_list = []
    for audio_name in audio_names:
        print('opening ' + audio_name)
        if os.path.exists(audio_name) and audio_name.endswith("wav"):
            audio_list.append(AudioSegment.from_wav(audio_name))
    return audio_list        

def get_load_raw_audio_thalon_length():
    len_activates = len(os.listdir("./thalon_audios/filtered"))
    len_backgrounds = len(os.listdir("./database/wav_resampling/backgrounds_10sec"))
    len_negatives = len(os.listdir("./database/wav_resampling/general_lt_1sec"))
    return len_activates, len_negatives, len_backgrounds