import telegram
import os
import subprocess
import time

from config import API_TOKEN

WAV_FILE_NAME = 'wav_file_name.wav'
DOWNLOAD_FILE_NAME = 'voice.ogg'

bot = telegram.Bot(token=API_TOKEN)
ln = 0

try:
    while True:
        updates = bot.get_updates()
        if len(updates) > ln:
            ln = len(updates)

            message = updates[-1].message
            bot.get_file(file_id=message.voice.file_id).download(DOWNLOAD_FILE_NAME)
            chat_id = message.chat_id

            subprocess.call([
                'ffmpeg-20180526-63c4a4b-win64-static/bin/ffmpeg.exe',
                '-i', DOWNLOAD_FILE_NAME, WAV_FILE_NAME
            ])

            p = subprocess.Popen(
                ["D:\Program\\anaconda2\python.exe", "model_usage.py",
                 "--wav=" + WAV_FILE_NAME], stdout=subprocess.PIPE
            )
            output = p.stdout.read()
            bot.send_message(chat_id=chat_id, text=output.split()[-1])
        time.sleep(1)
except Exception as e:
    print e
finally:
    if os.path.isfile(DOWNLOAD_FILE_NAME):
        os.remove(DOWNLOAD_FILE_NAME)
    if os.path.isfile(WAV_FILE_NAME):
        os.remove(WAV_FILE_NAME)