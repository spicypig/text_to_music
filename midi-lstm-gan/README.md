# MIDI File LSTM & GAN
The code is based on the blog post[here](https://medium.com/@abrahamkhan/generating-pokemon-inspired-music-from-neural-networks-bc240014132)! 

## Requirements:
* Python 3.x
* Installation of the following packages:
    * Music21
    * Keras
    * Tensorflow

## GAN
For creating a GAN to generate music, run mlp_gan.py. This will parse all of midi files and train a Conditional GAN model on them, with an Bi-directional LSTM-based discriminator and generator. After training, the generator will be fed random noise and a chosen emotion to make an output that will be converted into a .mid file using Music21. A plot of the discriminator and generator loss will also be saved, as well as the model architecture.
