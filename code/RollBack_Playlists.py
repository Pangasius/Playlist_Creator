import os
import shutil

wav_dir = "../ALL_SONGS/wav/"
mp3_dir = "../ALL_SONGS/mp3/"
end_dir = "../ALL_SONGS/end/"

for subdir, dirs, files in os.walk(end_dir):
    for file in files:
        path = os.path.join(subdir, file)
        
        if path.endswith(".wav"):
            shutil.move(path, wav_dir)
        
        elif path.endswith(".mp3") :
            shutil.move(path, mp3_dir)
                