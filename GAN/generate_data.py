import csv
import glob
import numpy as np
import os
import keras

from music21 import converter, instrument, note, chord
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.utils import to_categorical

# load music samples
def load_real_samples():
    (trainX, trainy) = load_dataset()
    print("load_real_samples", trainX.shape, trainy.shape)
    # convert from ints to floats
    X = trainX.astype('float32')
    return [X, trainy]

# # select real samples
def generate_real_samples(dataset, n_samples, vocab_size):
    # split into music notes and labels
    notes, labels = dataset
    onehot_encoded_notes = to_categorical(notes, num_classes=vocab_size)
    # choose random instances
    ix = randint(0, notes.shape[0], n_samples)
    # select notes and labels
    X, labels = onehot_encoded_notes[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, vocab_size, n_classes=10):
    # generate points in the latent space
    x_input = randint(vocab_size, size=(n_samples, latent_dim)).astype('float32')
    print("x_input: ", x_input)
    # 64, 10, 50
    z_input = to_categorical(x_input, num_classes=vocab_size)
    # generate labels
    labels = randint(0, n_classes, n_samples)
    #(64, 10, 50) (64,)
    return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples, int_to_note, vocab_size):
    # generate points in latent space
    z_input, labels_input = generate_latent_points(latent_dim, n_samples, vocab_size)
    # predict outputs
    predict_outputs = generator.predict([z_input, labels_input])
    # (64, 10)
    int_output = np.asarray(np.argmax(predict_outputs, axis=2))
    # (64, 10, 50)
    d_input = to_categorical(int_output, num_classes=vocab_size)
    # create class labels
    y = zeros((n_samples, 1))
    return [d_input, labels_input], y


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

# Return int_to_note mapping and vocab_size
def create_int_to_note_mapping():
    int_to_note = {}
    pitchnames = []

    for index, notes in get_notes().items():
        for note in notes:
            pitchnames.append(note)

    pitchnames = sorted(set(pitchnames))

    for number, note in enumerate(pitchnames):
        int_to_note[number] = note
  
    return int_to_note, len(int_to_note);

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