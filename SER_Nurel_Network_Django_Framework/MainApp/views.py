from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
#importing modules for prediction 
import librosa
import soundfile
import numpy as np


def hoempage(request):
    context = {'a':1}
    return render(request,'homepage.html',context)

def mic(request):
    context = {'a':1}
    return render(request,'mic.html',context)

def record(request):
    context = {'a':1}
    return render(request,'mic.html',context)

def file(request):
    context = {'a':1}
    return render(request,'file.html',context)

def content(request):
    context = {'a':1}
    return render(request,'content.html',context)

def history(request):
    return render(request,'history.html')


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
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
        '''if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))'''
    return result

def predictEmotion(request):
    import joblib
    import os
    file = '../SER_NUREL_NETWORK_DJANGO_FRAMEWORK/savedModels/model.joblib'
    load_model= joblib.load(file)
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['audioFile']
    
    print(fileObj)
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    print(type(filePathName),filePathName)
    fe = extract_feature('./'+filePathName,True,True,True)
    #DataFlair - Predict for the test set
    y_pred=load_model.predict([fe])
    context={'emotion':y_pred}
    return render(request,'result.html',context)
