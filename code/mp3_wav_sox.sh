for i in ../ALL_SONGS/mp3/*.mp3;do j=$(cut --complement -d "." -f 2- <<< $(cut -d "/" -f 4- <<< $i));sox --bits 32 --channels 2 --encoding signed-integer --rate 44100 "$i" ../ALL_SONGS/wav/"$j".wav; done;