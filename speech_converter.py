import pyttsx3

class SpeechConverter:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume level
        
    def convert_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()
