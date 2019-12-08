import conditional_gan
import generate_data

# size of the latent space
latent_dim = 10
int_to_note, vocab_size = generate_data.create_int_to_note_mapping()
print("int_to_note: ", int_to_note)
print("vocab_size is: ", vocab_size)
# load music data
dataset = generate_data.load_real_samples()
# create the generator
g_model = conditional_gan.define_generator(latent_dim, vocab_size)
# create the discriminator
d_model = conditional_gan.define_discriminator(latent_dim, vocab_size)
# create the gan
gan_model = conditional_gan.define_gan(g_model, d_model)

# train model
conditional_gan.train(g_model, d_model, gan_model, dataset, latent_dim, int_to_note, vocab_size, 100)
