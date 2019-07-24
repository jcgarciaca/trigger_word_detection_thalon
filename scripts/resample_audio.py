import librosa
import os

src_folder = '/home/JulioCesar/trigger_word_project/database/join_backgrounds'
dst_folder = '/home/JulioCesar/trigger_word_project/database/wav_resampling/backgrounds'

audio_list = os.listdir(src_folder)

for audio_name in audio_list:
    src = os.path.join(src_folder, audio_name)
    dst = os.path.join(dst_folder, audio_name)
    y, s = librosa.load(src, sr = 44100)
    librosa.output.write_wav(dst, y, sr = 44100)