# CS230 Project: Literary Muzak

## Prerequisites

> __Below we assume the working directory is the repository root.__

### Install dependencies

- Create a virtual environment

  ```sh
  conda create -n text_to_music python=3.4
  ```

- Using pip

  ```sh
  # Install the dependencies
  pip install -r requirements.txt
  ```

## Datasets

The initial dataset we used to jump start the emotion to music generation model is [Music and Emotion Datasets](https://doi.org/10.7910/DVN/IFOBRN) from Harvard Dataverse. There are 200 .wav sounds corresponding to the design matrix. The design matrix has a Melody column has 4 categories of emotions for the corresponding music, i.e 1 = Sad, 2 = Happy, 3 = Scary, 4 = Peaceful.

We prepared the datasets by converting all .wav files in one folder to .mid (MIDI) suffix using the following script. The conversion tool it is based on [audio_to_midi_melodia](https://github.com/justinsalamon/audio_to_midi_melodia).  

  ```sh
  # WAV to midi conversion
  cd scripts
  ./wav_to_midi.sh
  ```

## End-to-end workflow

![end-to-end](./graphs/end_to_end_model.png)

## Text to emotion

Our current text to emotion model is based on [Multi-class Emotion Classification for Short Texts](https://github.com/tlkh/text-emotion-classification). The model uses "multi-channel" combinations of convolutional kernels (ala CNN) and Long Short-Term Memory (LSTM) units to classify short text sequences (in our case, tweets) into one of five emotional classes, as opposed to the typical binary (positive/negative) or ternary (positive/negative/neutral) classes. 

The model performance achieved a positive result by achieving more than 62% overall classification accuracy and precision. In particular, they have achieved good validation accuracy on happy, sad, hate and anger (91% precision!) classes.

We will improve the model performamce in the context of music as the future work. We are planning to use music lyrics to train the model as well.

## Emotion to music 

![music-to-emotion](./graphs/emotion_to_music.svg)

We train a C-GAN (Conditional GAN):
- Generator, with inputs an emotion E and white noise WN, will learn to generate emotion-styled music M from tweaking the noise, to fool the Discriminator into thinking it's real music with the emotion it claims to have.
- Discriminator, with input music M and emotion label E, will learn to tell two things:
  - Whether the music is real or fake.
  - Whether the emotion E matches the emotion from music M.
