import streamlit as st
import librosa
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Sequential, model_from_json
import subprocess



def cut_song(song):
  start = 0
  end = len(song)
  
  song_pieces = []

  offset = 50000
  while start + offset < end:
    song_pieces.append(song[start:start+offset])
    start += offset

  return song_pieces

def prepare_song(song_path,label=0): 
  list_matrices = []
  y,sr = librosa.load(song_path,sr=22050)
  song_pieces = cut_song(y)
  for song_piece in song_pieces:
    melspect = librosa.feature.melspectrogram(song_piece)
    list_matrices.append(melspect)
  return list_matrices

thisdict = {"1" : "Normal", "2" : "Murmur", "3" : "Artifact"}
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

def show_predict_page():
    st.title("Heartbeat Prediction")

    st.write("""### We need some information to predict the class""")
    file = st.file_uploader("Please choose a file")


    song_name = file
    ok = st.button("Make Prediction")

    record = st.button("Record..")
    if ok and file!=None:
        print(song_name)
        if song_name.type == 'audio/mpeg':
          with open(os.path.join("tempDir","tempname.mp3"),"wb") as f:
            f.write(song_name.getbuffer())
          subprocess.call(['ffmpeg', '-i', 'tempDir/tempname.mp3','tempDir/tempname.wav'])
          song_name = 'tempDir/tempname.wav'
        song_pieces = prepare_song(song_name)
        all_tracks = song_pieces
        genre = ([0]*len(song_pieces))

        pops = np.array(all_tracks)
        pops = pops.reshape(-1,128,98,1)
        predval = loaded_model.predict(pops)
        classcase = 0
        predicted = [0,0,0]
        for i in range(len(predval)):
            arr = predval[i]
            index = np.where(arr == np.amax(arr))
            predicted[index[0][0]]+=1
        for i in range(3):
            ff = (100*predicted[i])/len(predval)
            ff = (round(ff, 2))
            if ff>=50 : 
                classcase = i+1
            print("Class " + str(i+1) + " Probab = " + str(ff) + "%")
            st.subheader("Class " + thisdict[str(i+1)] + " Probab = " + str(ff) + "%")

        # salary = regressor.predict(X)
        st.subheader(f"The estimated class is : " + thisdict[str(classcase)])
        for filename in os.listdir('tempDir'):
          fook = os.path.join('tempDir', filename)
          os.remove(fook)
    elif file:
      st.subheader(f"FILE UPLOADED")
    elif ok and file==None:
      st.subheader(f"Please ADD FILE!!!")
    else :
      st.subheader("Upload file..")
  
      


show_predict_page()