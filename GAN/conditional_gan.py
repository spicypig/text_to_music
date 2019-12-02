# example of training an conditional gan on the fashion mnist dataset
import csv
import keras
import numpy as np
import glob
import os
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from music21 import converter, instrument, note, chord

# Returns:
# @song_index_to_notes: music index to notes mappings.
def get_notes():
    """
    Get all the notes and chords from the midi files in the ../data/midi/ directory
    Create a list of notes for each song.
    """
    song_index_to_notes = {}

    for file in glob.glob("../data/midi/*.mid"):
        midi = converter.parse(file)
        song_index = int(os.path.splitext(os.path.basename(file))[0])
        print("Parsing %s with an index %d" % (file, song_index))

        notes_to_parse = None
        notes = []
        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        song_index_to_notes[song_index] = notes

    return song_index_to_notes


# Returns:
# @song_index_to_emotion: music index (int) to emotion mapping.def get_emotions():
def get_emotions():
    """ Read the design matrix csv file, returns a mapping from file name to emotions"""
    with open('../data/design_matrix.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        song_index_to_emotion = {}
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            print(f'Music file {row["Nro"]} is with mode {row["Mode"]}.')
            line_count += 1
            song_index_to_emotion[int(row["Nro"])] = row["Mode"]

        print(f'Processed {line_count} lines.')
        print("song_index_to_emotion size: ", len(song_index_to_emotion))

        return song_index_to_emotion;

# Generate input from real data to the discriminator
# Input:
#   @sequence_length: put the length of each sequence to be a default 10 notes/chords
# Returns:
#   @(train_x, train_y): music notes to emotion mappings, with 'sequence_length' per
#     per training example.
def load_dataset(sequence_length=10):
    """ Prepare the datasets used by the Neural Network """
    train_x = []
    train_y = []
    notes_to_emotion = []
    song_index_to_notes = get_notes()
    song_index_to_emotion = get_emotions()

    for index, notes in song_index_to_notes.items():
        if index in song_index_to_emotion:
            notes_to_emotion.append((notes, song_index_to_emotion[index]))

    for notes, emotion in notes_to_emotion:
        # get all pitch names
        pitchnames = sorted(set(item for item in notes))

        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        for i in range(0, int(len(notes) / sequence_length)):
            music_in = notes[i * sequence_length: (i + 1) * sequence_length]
            train_x.append([note_to_int[char] for char in music_in])
            train_y.append(emotion)

    print("train_x has shape: ", len(train_x))
    print("train_y has shape: ", len(train_y))

    return (np.asarray(train_x), np.asarray(train_y))

# define the standalone discriminator model
# @network_input: music input
# @n_classes: number of emotion categories
def define_discriminator(latent_dim, n_classes=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input, 10 can be tuned
    li = Embedding(n_classes, 10)(in_label)
    # music generator input
    in_music = Input(shape=(1, latent_dim))
    # merge music gen and label input
    merge = Concatenate()([in_music, li])
    # LSTM
    fe = LSTM(
        256, # units, dimensionality of the output space.
        input_shape=merge.shape,
        return_sequences=True
    )(merge)
    fe = Dropout(0.3)(fe)
    # downsample
    fe = LSTM(512, return_sequences=True)(fe)
    # dropout
    fe = Dropout(0.3)(fe)
    # LSTM
    fe = LSTM(256)(fe)
    # Dense
    fe = Dense(256)(fe)
    # dropout
    fe = Dropout(0.4)(fe)
    # output
    out_layer = Dense(1, activation='sigmoid')(fe)
    # define model
    model = Model([in_music, in_label], out_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# define the standalone generator model
# @latent_dim: input dimension of the noise
# @output_dim: output dimension of the generated music
# Simple network consisting of three LSTM layers,
# three Dropout layers, two Dense layers and one activation layer
def define_generator(latent_dim, output_dim=10, n_classes=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 10)(in_label)
    # music generator input
    in_music = Input(shape=(1, latent_dim))
    # merge music gen and label input
    merge = Concatenate()([in_music, li])
    print(merge.shape)
    # LSTM
    gen = LSTM(
        256, # units, dimensionality of the output space.
        input_shape=merge.shape,
        return_sequences=True
    )(merge)
    gen = Dropout(0.3)(gen)
    # LSTM
    gen = LSTM(512, return_sequences=True)(gen)
    # dropout
    gen = Dropout(0.3)(gen)
    # LSTM
    gen = LSTM(256, return_sequences=True)(gen)
    # The key for many-to-many LSTM:
    # TimeDistributed adds an independent layer for each time step in the recurrent model.
    # So, for instance, if we have 10 time steps in a model, a TimeDistributed layer
    # operating on a Dense layer would produce 10 independent Dense layers,
    # one for each time step. The activation for these dense layers is set to be softmax
    # in the final layer of our Keras LSTM model.
    gen = TimeDistributed(Dense(output_dim))(gen)
    out_layer = Activation('softmax')(gen)
    # define model
    model = Model([in_music, in_label], out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_label = g_model.input
    # get music output from the generator model
    gen_output = g_model.output
    # connect music output and label input from generator as inputs to discriminator
    gan_output = d_model([gen_output, gen_label])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_label], gan_output)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

# load music samples
def load_real_samples():
    (trainX, trainy) = load_dataset()
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1).reshape((-1, 1, 10))
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    # X = (X - 127.5) / 127.5
    return [X, trainy]

# # select real samples
def generate_real_samples(dataset, n_samples):
    # split into music notes and labels
    notes, labels = dataset
    # choose random instances
    ix = randint(0, notes.shape[0], n_samples)
    # select notes and labels
    X, labels = notes[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=10):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    songs = generator.predict([z_input, labels_input])
    # create class labels
    y = zeros((n_samples, 1))
    return [songs, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            print(X_real.shape, labels_real.shape, y_real.shape)
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save('cgan_generator.h5')

# size of the latent space
latent_dim = 10
# create the discriminator
d_model = define_discriminator(latent_dim)
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load music data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
