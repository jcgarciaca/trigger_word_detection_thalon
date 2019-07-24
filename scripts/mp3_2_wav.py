import os
from pydub import AudioSegment

src_folder = '/home/JulioCesar/trigger_word_project/database/general_mp3/clips'
dst_folder = '/home/JulioCesar/trigger_word_project/database/general_wav'

audio_list = os.listdir(src_folder)

for audio_name in audio_list:
    src = os.path.join(src_folder, audio_name)
    dst = os.path.join(dst_folder, audio_name.split('.')[0] + '.wav')
    print(src, ' -->', dst)
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")