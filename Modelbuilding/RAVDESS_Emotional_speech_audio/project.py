import librosa
import soundfile
import os, glob, pickle
import pyttsx3
import numpy as np
import pyaudio
import numpy
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



"""
Here,I have defined a function extract_feature to extract the 
mfcc, chroma, and mel features from a sound file. 
This function takes 4 parameters- the file name 
and three Boolean parameters for the three features:

"""

import warnings
warnings.filterwarnings("ignore")


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
            return result

"""
Here, I defined a dictionary to hold numbers 
and the emotions available in the RAVDESS dataset

"""
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
speaker = pyttsx3.init()
speaker.setProperty("rate", 180)


listener = sr.Recognizer()
def talk(text):
    speaker.say(text)
    speaker.runAndWait()

# Here, I created a list of emotions that has to be detected from the voice.p

observed_emotions=[ 'happy', 'surprised','sad']

"""
Here, I created a fuction load_data() – this takes in the relative 
size of the test set as parameter. x and y are empty lists; 
I used the glob() function from the glob module to
get all the pathnames for the sound files in our dataset. 
Here I used the Ravdess data set which contain 24 folders of different 
voice with 60 audio files in each folder

"""

def get_info() :      #function for geeting voice from user and then  converting it into the text
    with sr.Microphone() as source:
        print('listening...')
        voice = listener.record(source,duration=4)

        try:
            info = listener.recognize_google(voice)
            #print(format(info))
            return info.lower()

        except:
            talk('did not hear properly please speak loudly')
            get_info()


def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)




talk("if you want to test the model on file from system speak system else speak live for recording your audio");

option=get_info()
print(option)

if option != "live":
    talk("i had taken a file from your system wait for results")
    file="Actor_01/03-01-01-01-01-01-01.wav"
    
else:
    talk("start speaking something I am ready to record your audio")
    RATE=16000
    RECORD_SECONDS = 15
    CHUNKSIZE = 1024
    # initialize portaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
    frames = [] # A python-list of chunks(numpy.ndarray)
    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
     data = stream.read(CHUNKSIZE)
     frames.append(numpy.fromstring(data, dtype=numpy.int16))
    #Convert the list of numpy-arrays into a 1D array (column-wise)
    numpydata = numpy.hstack(frames)
    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    import scipy.io.wavfile as wav
    wav.write('out.wav',RATE,numpydata)
    talk("your audio is recorded wait for some time to get results")
    file="out.wav"



    
    



feature=extract_feature(file,mfcc=True,chroma=True,mel=True)





x_train,x_test,y_train,y_test=load_data(test_size=0.2)





# print((x_train.shape[0], x_test.shape[0]))



# print(f'Features extracted: {x_train.shape[1]}')



model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pre=model.predict([feature])
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)


#print("Accuracy: {:.2f}%".format(accuracy*100))



"""
Here I have taken a varible speaker to use it as speaker 
which speak our obsereved emotion and accuracy of the model

"""
speaker = pyttsx3.init()


"""
Here I have created funtion a talk which is used to talking purpose
in the code we just have to pass the line or whatever we want to speak 

"""




print(y_pre)
talk(' the obeserved emotion from the given audio file is ')
talk(y_pre)
talk(' and the accuracy of detection of emotion is  ')
accu=int(accuracy*100)
print(accu)
talk(accu)
talk('%')