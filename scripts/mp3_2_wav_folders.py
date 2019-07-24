import os
from pydub import AudioSegment

src_folder = '/home/JulioCesar/trigger_word_project/database/backgrounds'
dst_folder = '/home/JulioCesar/trigger_word_project/database/join_backgrounds'

subfolders_list = os.listdir(src_folder)

for subfolder in subfolders_list:
    audio_list = os.listdir(os.path.join(src_folder, subfolder))
    for audio_name in audio_list:
        src = os.path.join(src_folder, subfolder, audio_name)
        dst = os.path.join(dst_folder, audio_name.split('.')[0] + '.wav')
        index = 0
        while os.path.exists(dst):
            dst = os.path.join(dst_folder, audio_name.split('.')[0] + '_' + str(index) + '.wav')
            index += 1
        print(src, ' -->', dst)
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")