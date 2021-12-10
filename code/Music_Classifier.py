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

#EXTERNAL CUSTOM IMPORTS
import RollBack_Playlists

global wav_dir, mp3_dir, end_dir, from_dir
wav_dir = "../ALL_SONGS/wav/"
mp3_dir = "../ALL_SONGS/mp3/"
end_dir = "../ALL_SONGS/end/"

def common_data(time) :
    freq = 44100
    sampling = freq * 30
    short = 245
    
    return freq, sampling, short


def load_data(features_length, n_jobs, time) :
    executor = ThreadPoolExecutor(n_jobs)

    for subdir, dirs, files in os.walk(from_dir):
        number_samples = 0
        
        for file in files:
            number_samples += 1
      
        music_samples = np.zeros([number_samples, features_length])
        music_names = [None]*number_samples

        file_num = 0
        
        futures = [None]*number_samples
            
        for file in files:
            
            music_names[file_num] = file[:-4]
            
            audio_path = [os.path.join(subdir, file)]
            
            if (not audio_path[0].endswith(".wav")) and (not audio_path[0].endswith(".mp3")) :
                continue
            
            try :
                futures[file_num] = executor.submit(__load_and_extract, copy.copy(audio_path)[0], features_length,\
                                                    music_samples, file_num, number_samples, time)
            except :
                print("\n ! \n Exception raised : could not launch in parallel")
                traceback.print_exc()
                print("\n ! \n")
                
            file_num += 1
            
        executor.shutdown(wait=True)
        
        return music_names, music_samples

def __load_and_extract(audio_path, features_length, music_samples, file_num, number_samples, time) :
    
    freq, sampling, short = common_data(time)
    
    print("launched " + str(file_num), end='\r')
        
    try :
        if audio_path.endswith(".wav") :
            x , freq = librosa.load(audio_path, sr=freq, offset=40, duration=time)
        else :
            print("Invalid format for " + audio_path)
            return

        features = np.zeros([features_length])

        if  features_length == short :
            spectral_centroids = librosa.feature.spectral_centroid(x, sr=freq)[0]
            centroids = np.reshape(np.array(pd.DataFrame(spectral_centroids).describe()[1:]), [7])
            features[0:7] = centroids / np.linalg.norm(centroids)
            del centroids, spectral_centroids
            
            mfccs = librosa.feature.mfcc(x, sr=freq)
            mf = np.reshape(np.array(pd.DataFrame(np.transpose(mfccs)).describe()[1:]), [140])
            features[14:154] = mf / np.linalg.norm(mf)
            del mf, mfccs
            
            spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=freq)[0]
            roll_off = np.reshape(np.array(pd.DataFrame(spectral_rolloff).describe()[1:]), [7])
            features[154:161] = roll_off / np.linalg.norm(roll_off)
            del roll_off, spectral_rolloff

            hop_length = 512
            
            chromagram = librosa.feature.chroma_stft(x, sr=freq, hop_length=hop_length)
            chro = np.reshape(np.array(pd.DataFrame(np.transpose(chromagram)).describe()[1:]), [84])
            features[161:245] = chro / np.linalg.norm(chro)
            del chro, chromagram
        else :
            print("No method specified")
            exit()
       
        #print("Currently at : " + str(file_num) + " out of " + str(number_samples) + "\t", end='\r')
        
        music_samples[file_num] = features.copy()
        
        return
    except :
        print("\n ! \n Exception raised : futures ")
        traceback.print_exc()
        print("\n ! \n")
        
def clean() :
    shutil.rmtree(end_dir)
    os.mkdir(end_dir)
    return
    
def import_songs() :
    process = subprocess.Popen(["bash mp3_wav_sox.sh"], shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()
    
    for lines in out :
        print(out)
    return
        
def dendrogram_create(music_data, music_names, method):
    Z = linkage(music_data, 'ward', optimal_ordering=True)
    
    fig, axes = plt.subplots(1,1)
    dendrogram(Z,  labels=music_names, leaf_rotation=0,\
               orientation="left", color_threshold='default', \
               above_threshold_color='grey', leaf_font_size=2)
    axes.set_title("Dendrogram of " + str(len(music_names)) + " musics out of 6 playlists on youtube")
    axes.set_ylabel("Relative distance")
    
    plt.savefig(method + ".pdf", bbox_inches='tight')
    return

def preshow(music_data, music_names, number_genres) :
    labels = [None] * 2
    
    labels[0] = k_means(X = music_data, n_clusters = number_genres)[1]
    labels[1] = Birch(n_clusters=number_genres, threshold=np.mean(music_data[:,0])).fit(music_data).labels_

    sep = 0
    print("\n\n\nK_means sorting : \n\n\n")
    for pairs in sorted(zip(music_names,np.transpose(labels)), key=lambda x: x[1][0]) :
        print(pairs[0])
        if pairs[1][0] != sep :
            print("--------")
            sep = pairs[1][0]
    
    sep = 0
    print("\n\n\nBirch sorting : \n\n\n")
    for pairs in sorted(zip(music_names,np.transpose(labels)), key=lambda x: x[1][1]) :
        print(pairs[0])
        if pairs[1][1] != sep :
            print("--------")
            sep = pairs[1][1]
            
    out = -1
    while out != 0 and out != 1 :
        out = int(input("Type 0 for k-means, 1 for birch"))
    return labels[out]

def move_all(labels, music_names, alter_path, move) :
    for pairs in sorted(zip(music_names,labels), key=lambda x: x[1]) :
        from_path = from_dir + str(pairs[0]) + "." + alter_path[:3]
        to_dir = end_dir + "/" + str(pairs[1])
        to_path =  to_dir + "/" + str(pairs[0]) + "." + alter_path[:3]
        
        if not os.path.exists(to_dir) :
            os.mkdir(to_dir)

        if move == "move" :
            shutil.move(from_path, to_path)
        else :
            shutil.copy(from_path, to_path)
    return
        
def main(method="short", number_genres=5, alter_path="wav", move="move", n_jobs=3, time=30, dendo="dendo") :    
    for i in range (1, len(sys.argv)) :
        if sys.argv[i].startswith("method=") :
            method = sys.argv[i].split('=')[1]
        elif sys.argv[i].startswith("number_genres=") :
            number_genres = int(sys.argv[i].split('=')[1])
        elif sys.argv[i].startswith("alter_path=") :
            alter_path = sys.argv[i].split('=')[1]
        elif sys.argv[i].startswith("move=") :
            move = sys.argv[i].split('=')[1]
        elif sys.argv[i].startswith("n_jobs=") :
            n_jobs = int(sys.argv[i].split('=')[1])
        elif sys.argv[i].startswith("time=") :
            time = int(sys.argv[i].split('=')[1])
        elif sys.argv[i].startswith("dendo=") :
            dendo = sys.argv[i].split('=')[1]
            
    freq, sampling, short = common_data(time)
    
    global wav_dir, mp3_dir, end_dir, from_dir
    
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
        
    if (not os.path.exists(wav_dir)) or  (not os.path.exists(mp3_dir)) or (not os.path.exists(end_dir)):
        print("yo yo, make sure all songs are in ../ALL_SONGS/mp3, and that there exists ../ALL_SONGS/wav and ../ALL_SONGS/end")
        exit()

    if method == "short" :
        features_length = short
    else :
        print("Parameter one should be default or short")
        exit()
    
    if alter_path == "wav_import" :
        print("Importing songs to .wav")
        import_songs()
        from_dir = wav_dir     
    elif alter_path == "wav" :
        from_dir = wav_dir     
    elif alter_path == "mp3" :
        if n_jobs != 1 :
            print("Unfortunately, mp3 format is slow and doesn't support multithreading")
            print("Defaulted to n_jobs = 1")
            n_jobs = 1
        from_dir = mp3_dir
    
    print("Creating features and extracting them")
    
    music_names, music_data = load_data(features_length, n_jobs, time)
    
    print("\n Creating " + str(number_genres) + " genres")
    
    labels = preshow(music_data, music_names, number_genres)

    print("Copying/Moving .mp3/wav to new folders")
    move_all(labels, music_names, alter_path, move)
    
    try :
        if dendo == "dendo" : 
            print("Creating dendrogram")
            dendrogram_create(music_data, music_names, method)
    except :
        print("\n ! \n Exception raised : dendo")
        traceback.print_exc()
        print("\n ! \n")        
        
    print("All done ^^")
    return music_names, music_data

RollBack_Playlists
#clean()
music_names, music_data = main()
