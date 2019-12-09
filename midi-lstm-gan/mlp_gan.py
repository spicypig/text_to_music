from __future__ import print_function, division
import matplotlib.pyplot as plt
import csv
import sys
import numpy as np
import pickle
import glob
import keras
import os
from tensorflow.keras import backend
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from tensorflow.keras.utils import plot_model

# Returns note_to_emotion pairs
def read_note_from_file(filename, emotion):
    midi = converter.parse(filename)
    note_to_emotion = []
    print("Parsing %s" % filename)

    notes_to_parse = None

    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
        
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            note_to_emotion.append((str(element.pitch), emotion))
        elif isinstance(element, chord.Chord):
            note_to_emotion.append(('.'.join(str(n) for n in element.normalOrder), emotion))

    return note_to_emotion

def get_note_to_emotion():
    """ Get all the notes and chords from the midi files """
    note_to_emotion = []
    file_name_to_emotion = {}
    emotion_dict = get_song_index_to_emotion()
    # parse file with emotions.
    for file in glob.glob("../data/midi/*.mid"):
        song_index = int(os.path.splitext(os.path.basename(file))[0])
        if song_index in emotion_dict:
            file_name_to_emotion[file] = emotion_dict[song_index]
        
    # parse file without emotions. All happy music
    #for file in glob.glob("../data/Pokemon MIDIs/*.mid"):
    #   file_name_to_emotion[file] = 2
     
    # parse final fantasy songs, all peaceful music
    #for file in glob.glob("../data/final_fantasy_songs/*.mid"):
    #    file_name_to_emotion[file] = 4

    # parse pop songs
    #for file in glob.glob("../data/Pop_Music_Midi/*.midi"):
    #    file_name_to_emotion[file] = 0 
    
    # Read notes from files 
    for file, emotion in file_name_to_emotion.items():
        note_to_emotion += read_note_from_file(file, emotion)

    return note_to_emotion

# Returns:
# @song_index_to_emotion: music index (int) to emotion mapping.def get_emotions():
def get_song_index_to_emotion():
    """ Read the design matrix csv file, returns a mapping from file name to emotions"""
    with open('../data/design_matrix.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        song_index_to_emotion = {}
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            line_count += 1
            song_index_to_emotion[int(row["Nro"])] = int(row["Melody"])

        return song_index_to_emotion;

def prepare_sequences(note_to_emotion, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # Get all pitch names
    pitchnames = sorted(set(note for note, emotion in note_to_emotion))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(note_to_emotion) - sequence_length, 1):
        sequence_in = note_to_emotion[i:i + sequence_length - 5]
        sequence_out = note_to_emotion[i + sequence_length][0]
        new_input = []
        for note, emotion in sequence_in:
            new_input.append(note_to_int[note])
        # append emotion
        for note, emotion in sequence_in[:5]:
            new_input.append(emotion)
        network_input.append(new_input)
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize input between -1 and 1
    network_input = (network_input - float(n_vocab)/2) / (float(n_vocab)/2)
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_midi(prediction_output, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for item in prediction_output:
        pattern = item[0]
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='{}.mid'.format(filename))
    return output_notes

class GAN():
    def __init__(self, rows):
        self.seq_length = rows
        self.seq_shape = (self.seq_length, 1)
        self.latent_dim = 1000
        self.num_emotion = 4
        self.disc_loss = []
        self.gen_loss =[]

        # note and emotion sets
        self.note_to_emotion = get_note_to_emotion()

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates note sequences
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(generated_seq)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_generator(self):
        return self.generator

    def build_discriminator(self):

        model = Sequential()
        model.add(LSTM(512, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(512)))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)    
        seq = Input(shape=self.seq_shape)
        validity = model(seq)

        return Model(seq, validity)
      
    def build_generator(self):

        model = Sequential()
        model.add(LSTM(256, input_shape=self.seq_shape, return_sequences=True))
        model.add(Bidirectional(LSTM(256)))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.seq_shape), activation='tanh'))
        model.add(Reshape(self.seq_shape))
        model.summary()
        plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)    
        
        noise = Input(shape=self.seq_shape)
        seq = model(noise)

        return Model(noise, seq)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load and convert the data
        n_vocab = len(set([note for note, emotion in self.note_to_emotion]))
        X_train, y_train = prepare_sequences(self.note_to_emotion, n_vocab)
        print("vocab_sizes: ", n_vocab)
        print("X_train: ", X_train.shape)
        print("batch size: ", batch_size)

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Training the model
        for epoch in range(epochs):

            # Training the discriminator
            # Select a random batch of note sequences
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_seqs = X_train[idx]

            #noise = np.random.choice(range(484), (batch_size, self.latent_dim))
            #noise = (noise-242)/242
            noise = np.random.normal(0, 1, (batch_size, 1, self.latent_dim))
            # Generate a batch of new note sequences
            gen_seqs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_seqs, real)
            d_loss_fake = self.discriminator.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            #  Training the Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as real)
            g_loss = self.combined.train_on_batch(noise, real)

            # Print the progress and save into loss lists
            if epoch % sample_interval == 0:
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
              self.disc_loss.append(d_loss[0])
              self.gen_loss.append(g_loss)
        # save the generator model
        self.generator.save_weights('cgan_generator.h5')
        self.plot_loss()
        
    def generate(self, emotion, out_index):
        # Get pitch names and store in a dictionary
        pitchnames = sorted(set(note for note, emotion in self.note_to_emotion))
        # Create a dictionary to map pitches to integers
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        # Use random noise to generate sequences
        music_noise = np.random.normal(0, 1, (1, self.latent_dim - 10))
        
        noise = np.concatenate((music_noise, [emotion] * 10), axis=None).reshape(1, self.latent_dim)
        predictions = self.generator.predict(noise)
        
        pred_notes = [x*242+242 for x in predictions[0]]
        pred_notes = [int_to_note[int(x)] for x in pred_notes]
        
        return create_midi(pred_notes, 'results/gan_final_' + str(emotion) + "_" + str(out_index))
        
    def plot_loss(self):
        plt.plot(self.disc_loss, c='red')
        plt.plot(self.gen_loss, c='blue')
        plt.title("GAN Loss per Epoch")
        plt.legend(['Discriminator', 'Generator'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('GAN_Loss_per_Epoch_final.png', transparent=True)
        plt.close()

if __name__ == '__main__':
  gan = GAN(rows=100)
  gan.train(epochs=100, batch_size=32, sample_interval=1)
  for i in range(10):
      for j in range(4):
          gan.generate(j + 1, i + 1)


