from pydub import AudioSegment
import os

src_folder = '/home/JulioCesar/trigger_word_project/database/wav_resampling/backgrounds'
dst_folder = '/home/JulioCesar/trigger_word_project/database/wav_resampling/backgrounds_10sec'

audios_list = os.listdir(src_folder)

sample = 10000 # 10 seconds

for number, audio_name in enumerate(audios_list):
    # if number > 2:
    #     break
    
    src = os.path.join(src_folder, audio_name)
    fullAudio = AudioSegment.from_wav(src)
    
    max_value = len(fullAudio)
    print(src, max_value)

    t1 = 0
    t2 = sample

    index = 0
    
    while True:       
        dst = os.path.join(dst_folder, audio_name.split('.')[0] + '_' + str(index) + '.wav')
        while os.path.exists(dst):
            dst = os.path.join(dst_folder, audio_name.split('.')[0] + '_' + str(index) + '.wav')
            index += 1

        newAudio = fullAudio[t1:t2]
        newAudio.export(dst, format="wav")
        print(dst + ' created')

        t1 += sample
        t2 += sample

        if t2 > max_value:
            break
    
    