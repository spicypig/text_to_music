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
from keras import backend as K
from music21 import converter, instrument, note, chord

# If activation is set to True, do activation with softmax.
class CustomDense(Dense):
    def __init__(self, units, **kwargs):
        super(CustomDense, self).__init__(units, **kwargs)

    def call(self, inputs):
        output = K.dot(inputs, K.softmax(self.kernel, axis=-1))
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation:
            output = self.activation(output)
        return output

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
        #print("Parsing %s with an index %d" % (file, song_index))

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
            #print(f'Music file {row["Nro"]} is with mode {row["Melody"]}.')
            line_count += 1
            song_index_to_emotion[int(row["Nro"])] = row["Melody"]

        #print(f'Processed {line_count} lines.')
        #print("song_index_to_emotion size: ", len(song_index_to_emotion))

        return song_index_to_emotion;

def create_int_to_note_mapping():
    int_to_note = {}
    song_index_to_notes = get_notes()
    song_index_to_emotion = get_emotions()
    notes_to_emotion = []

    for index, notes in song_index_to_notes.items():
        if index in song_index_to_emotion:
            notes_to_emotion.append((notes, song_index_to_emotion[index]))

    for notes, emotion in notes_to_emotion:
    	 # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        for number, note in enumerate(pitchnames):
            int_to_note[number] = note

    return int_to_note, len(int_to_note);

def get_vocab_size():
    vocab_size = len(create_int_to_note_mapping().keys())
    print("vocab_size is: ", vocab_size)
    return vocab_size;

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
        for i in range(0, int(len(notes)) - sequence_length):
            music_in = notes[i: i + sequence_length]
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
def define_generator(latent_dim, vocab_size, n_classes=10, output_dim=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, 10)(in_label)
    # music generator input
    in_music = Input(shape=(1, latent_dim))
    # merge music gen and label input
    merge = Concatenate()([in_music, li])
    # print(merge.shape)
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
    # don't need this unless we have > 1 time steps.
    # gen = TimeDistributed(Dense(1000))(gen)
    gen = Dense(output_dim * vocab_size)(gen)
    gen = Reshape((output_dim, vocab_size))(gen)
    out_layer = CustomDense(1, use_bias=False, activation='softmax')(gen)
    out_layer = Reshape((1, output_dim))(out_layer)
    # define model
    print("out_layer: ", out_layer.shape)
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
    print("load_real_samples", trainX.shape, trainy.shape)
    # expand to 3d, e.g. add channels
    X = expand_dims(trainX, axis=-1).reshape((-1, 1, 10))
    # convert from ints to floats
    X = X.astype('float32')
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
# TODO: this function has bugs.
def generate_latent_points(latent_dim, n_samples, vocab_size, n_classes=10):
    # generate points in the latent space
    x_input = np.random.randint(vocab_size, latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = x_input.reshape(n_samples, -1, latent_dim)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, int_to_note, vocab_size):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, vocab_size)
    # predict outputs
    predict_outputs = generator.predict([z_input, labels_input])
    print("predict_outputs: ", predict_outputs)
    int_output = np.asarray([np.argmax(output) for output in predict_outputs]);

    print(int_output.shape)
    # int_output = expand_dims(int_output, axis=-1).reshape((-1, 1, latent_dim))

    # create class labels
    y = zeros((n_samples, 1))
    return [int_output, labels_input], y

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, int_to_note, vocab_size, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            print("real: ", X_real.shape, labels_real.shape, y_real.shape)
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch, int_to_note, vocab_size)
            print("fake: ", X_fake.shape, labels.shape, y_fake.shape)
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
    g_model.save_weights('cgan_generator.h5')

# size of the latent space
latent_dim = 10
int_to_note, vocab_size = create_int_to_note_mapping()
print("int_to_note: ", int_to_note)
print("vocab_size is: ", vocab_size)

# load music data
dataset = load_real_samples()
# create the generator
g_model = define_generator(latent_dim, vocab_size)
# create the discriminator
d_model = define_discriminator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)

# train model
train(g_model, d_model, gan_model, dataset, latent_dim, int_to_note, vocab_size, 100)
