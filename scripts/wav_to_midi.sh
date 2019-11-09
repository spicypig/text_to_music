FILES=../data/music/*
for f in $FILES
do
  echo "Processing $f "
  # take action on each file. $f store current file name
  filename=$(basename -a -s .wav $f)
  python ../../audio_to_midi_melodia-master/audio_to_midi_melodia.py $f ../data/midi/$filename.mid 60 --smooth 0.25 --minduration 0.1

  output "../data/midi/$filename.mid" 
done
