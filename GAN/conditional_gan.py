import keras
import numpy as np
import generate_data
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
from keras.layers import Activation
from keras.utils import to_categorical
from keras import backend as K


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

# define the standalone discriminator model
# @network_input: music input
# @n_classes: number of emotion categories
def define_discriminator(latent_dim, vocab_size, n_classes=10):
    # label input
    in_label = Input(shape=(1,))
    # embedding for categorical input
    li = Embedding(n_classes, latent_dim)(in_label)
    li = Reshape((latent_dim, 1))(li)
    # music generator input
    in_music = Input(shape=(latent_dim, vocab_size))
    # merge music gen and label input
    merge = Concatenate(axis=2)([in_music, li])
    # (?, 10, 51)
    print("merge: ", merge.shape)
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
    li = Embedding(n_classes, latent_dim)(in_label)
    li = Reshape((latent_dim, 1))(li)
    # music generator input
    in_music = Input(shape=(latent_dim, vocab_size))
    # merge music gen and label input
    merge = Concatenate(axis=2)([in_music, li])
    # (?, 10, 51)
    print("merge: ", merge.shape)
    # LSTM
    gen = LSTM(
        256, # units, dimensionality of the output space.
        input_shape=merge.shape,
        return_sequences=True
    )(merge)
    gen = Dropout(rate=0.7)(gen)
    # LSTM
    gen = LSTM(512, return_sequences=True)(gen)
    # dropout
    gen = Dropout(rate=0.7)(gen)
    # LSTM
    # (?, ?, 256)
    gen = LSTM(256, return_sequences=True)(gen)
    gen = Dense(vocab_size)(gen)
    # 11, 50
    out_layer = CustomDense(50, use_bias=False, activation='softmax')(gen)
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

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, int_to_note, vocab_size, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            [X_real, labels_real], y_real = generate_data.generate_real_samples(dataset, half_batch, vocab_size)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            # generate 'fake' examples
            [X_fake, labels], y_fake = generate_data.generate_fake_samples(g_model, latent_dim, half_batch, int_to_note, vocab_size)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            # prepare points in latent space as input for the generator
            [z_input, labels_input] = generate_data.generate_latent_points(latent_dim, n_batch, vocab_size)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save_weights('cgan_generator.h5')
