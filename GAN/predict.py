import generate_data
import conditional_gan
import numpy as np

latent_dim = 10
int_to_note, vocab_size = generate_data.create_int_to_note_mapping()
print("int_to_note: ", int_to_note)
print("vocab_size is: ", vocab_size)

music_generator = conditional_gan.define_generator(latent_dim, vocab_size)
music_generator.load_weights('./cgan_generator.h5')


# use the generator to generate n fake examples, with class labels
def generate_music(generator, latent_dim, int_to_note, vocab_size):
    # generate points in latent space
    z_input, labels_input = generate_data.generate_latent_points(latent_dim, 1, vocab_size)
    # predict outputs
    predict_outputs = generator.predict([z_input, labels_input])
    print("predict_outputs: ", predict_outputs)
    # (64, 10)
    int_output = np.argmax(predict_outputs, axis=2).tolist()
    notes = []
    print("int_output: ", int_output[0])
    for int_note in int_output[0]:
      notes.append(int_to_note[int_note])
    return notes


notes = generate_music(music_generator, latent_dim, int_to_note, vocab_size)

print(notes)
#np.save('./song_output.npy', song_tuple[0][0])

