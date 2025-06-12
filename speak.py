from gtts import gTTS
tts = gTTS("Hello, I am a realistic male voice.", lang='en', tld='co.uk')  # UK male-sounding voice
tts.save("male_voice.mp3")
import os
os.system("mpg123 male_voice.mp3")
