# Exploring_VAE_Latent_Space
Explores and visualizes a VAE's latent space with you mouse and keyboard.

Creates latent dimensions from mouse position, decodes them using a VAE trained on MNIST using Louis Tiao's code, and visualizes them using Pygame.

All credit to Louis Tiao for the code to train the VAE (http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/).  Edit the code to save the encoder and decoder in seperate h5/hdf5 files, then load them into this folder.  To change the number of dimensions, edit `latent_dim` in Louis Tiao's original code.

Move cursor position on screen to change x and y values, and, if enabled, use left and right arrow keys to change z value in the latent dimension.
