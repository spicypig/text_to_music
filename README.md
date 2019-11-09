# CS230 Project: Automatic music composition from text

## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

- Using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

## Datasets

The initial dataset we used to jump start the emotion to music generation model is [Music and Emotion Datasets](https://doi.org/10.7910/DVN/IFOBRN) from Harvard Dataverse. There are 200 .wav sounds corresponding to the design matrix. The design matrix has a Melody column has 4 categories of emotions for the corresponding music, i.e 1 = Sad, 2 = Happy, 3 = Scary, 4 = Peaceful.

We prepared the datasets by converting all .wav files in one folder to .mid (MIDI) suffix using the following script. The conversion tool it is based on is [audio_to_midi_melodia](https://github.com/justinsalamon/audio_to_midi_melodia).  

  ```sh
  # WAV to midi conversion
  cd scripts
  ./wav_to_midi.sh 
  ```

## Text to sentimemts

## Sentiments to music 


