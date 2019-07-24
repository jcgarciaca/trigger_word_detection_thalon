import speech_recognition as sr
import os
import sys
from time import sleep

root_path = os.path.join(os.path.dirname(sys.path[0]), 'thalon_audios')
print(root_path)

audio_path = os.path.join(root_path, 'activate_')

r = sr.Recognizer()
mic = sr.Microphone()

counter = len(os.listdir(root_path)) + 1

while(True):
    # sleep(0.3)    
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print('****** Talk ******')
        audio = r.listen(source)
        with open(audio_path + str(counter) + '.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        data = raw_input('Audio completed. Press a key (x to exit): ')
        if data == 'x':
            break
        counter += 1
