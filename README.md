# Playlist_Creator
Tool to create playlists automatically


Hi, if you're using this tool there is a few things to know :

  ° Libraries used :
      
       #INTERNAL BASE IMPORTS
      import os
      import sys
      import copy
      import shutil
      import traceback
      import subprocess
      from concurrent.futures import ThreadPoolExecutor

      #EXTERNAL BASIC IMPORTS : DATA
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      from scipy.cluster.hierarchy import dendrogram, linkage

      #EXTERNAL BASIC IMPORTS : MUSIC
      import librosa

      #EXTERNAL BASIC IMPORTS : SORTING
      from sklearn.cluster import k_means 
      from sklearn.cluster import Birch


  ° Highly prefer .wav to .mp3, librosa doesn't support natively and takes ages to load it, even worse, it deadlocks if ran in parallel.
  
  
  -> I added a script named mp3_wav_sox.sh that uses SoX to export all .mp3 files from a director to .wav in another
  
  
  ° Highly recommand you keep the codes in a folder and have next to it folders as follows :
  
  
    - code
          *.py
          *.sh
    - ALL_SONGS
          -- end
          -- mp3
            *.mp3
          -- wav
            *.wav
            
            
  -> if you make any change be sure to change the code 
  
  
  ° Once the code is close to the end it will propose two sortings, choose 0/1 and input it, everything will be copied / moved to "end" in the correct playlists.
  
  
  ° Example run command : 
  
  
      From bash : python3 Music_Classifier.py method=short number_genres=4 alter_path=wav move=move n_jobs=3 time=15 dendo=dendo
      
      - method : [short] the distance metric used, currently only "short" available.
      - number_genres : [int] number of playlists that will be created /!\ depending on the run there can be 0 / 1 song in some plalists.
      - alter_path : [mp3, wav, wav_import], mp3 not recommended, wav fast and reliable, wav_import if you want to import all your mp3 to wav and run (needs SoX)
      - move : [move, None] if move: move, else copy.
      - n_jobs : [int] number of processes that will be run in parallel for the loading and distance calculations.
      - time : [int] number of seconds to read on the file beginning at 40 seconds, usually 15<x>30 does the trick
      - dendo : [dendo, None] if dendo : draw a dendrogram with all your songs once its done, can help choosing the optimal number of playlists.
      
      
 ° If you plan on changing the code, be careful not to erase all your songs forever with shutil.rmtree （＞人＜；）

Note on performance : With time set to 30 and 3 jobs, 115 wav songs takes about 2 minutes (+export if needed). (8GB RAM, I5) 
                      From mp3 I didn't have enough will to let it run all the way.
