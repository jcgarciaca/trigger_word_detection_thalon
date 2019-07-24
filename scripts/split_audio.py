from pydub import AudioSegment
import os

src_folder = '/home/JulioCesar/trigger_word_project/Keras-Trigger-Word/database/wav_resampling/general_lt_5sec'
dst_folder = '/home/JulioCesar/trigger_word_project/Keras-Trigger-Word/database/wav_resampling/general_lt_1sec'

audios_list = os.listdir(src_folder)

'''
t_1 = 0
t_2 = t_1 + 1
sample = 5000 # 5 seconds
'''

t1 = 2000
t2 = 3000

for audio_name in audios_list:
    src = os.path.join(src_folder, audio_name)
    dst = os.path.join(dst_folder, audio_name)
    newAudio = AudioSegment.from_wav(src)
    newAudio = newAudio[t1:t2]
    newAudio.export(dst, format="wav")
