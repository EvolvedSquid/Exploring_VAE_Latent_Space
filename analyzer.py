from keras.models import load_model, Model
from keras.layers import Layer
from keras.datasets import mnist
import keras.backend as K
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
from scipy.stats import norm
import random, math


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

vae = load_model('vae.hdf5', custom_objects={'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})
encoder = load_model('encoder_3d.hdf5', custom_objects={'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})
decoder = load_model('decoder_3d.hdf5', custom_objects={'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})
vae.summary()

original_dim = 784
intermediate_dim = 256
latent_dim = 2
batch_size = 100
epochs = 60
epsilon_std = 1.0

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_test = x_test.reshape(-1, original_dim) / 255

# # encoder = Model(vae.input[0], vae.get_layer('z_mu').output)

# # display a 2D plot of the digit classes in the latent space
# print(encoder.input_shape, x_test.shape)
# z_test = encoder.predict(x_test, batch_size=batch_size)
# plt.figure(figsize=(6, 6))
# plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
#             alpha=.4, s=3**2, cmap='inferno')
# plt.colorbar()
# plt.show()

# # display a 2D manifold of the digits
# n = 50  # figure with 15x15 digits
# digit_size = 28

# # linearly spaced coordinates on the unit square were transformed
# # through the inverse CDF (ppf) of the Gaussian to produce values
# # of the latent variables z, since the prior of the latent space
# # is Gaussian
# u_grid = np.dstack(np.meshgrid(np.linspace(0.05, 0.95, n),
#                                np.linspace(0.05, 0.95, n)))
# z_grid = norm.ppf(u_grid)
# x_decoded = decoder.predict(z_grid.reshape(n*n, 2))
# x_decoded = x_decoded.reshape(n, n, digit_size, digit_size)

# plt.figure(figsize=(10, 10))
# plt.imshow(np.block(list(map(list, x_decoded))), cmap='gray')
# plt.show()





import pygame


round_n = 3
dimensions = 3
width = height = 1008
w = h = 36
min_x = -2
max_x = 3
min_y = -2
max_y = 3
min_z = -2
max_z = 3

slider_z = 0.5

pygame.init()
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
done = False
clock = pygame.time.Clock()
while not done:
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        done = True
        keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()
        screen.fill((255, 255, 255))
        
        if keys[pygame.K_LEFT]:
            slider_z -= 0.05
        if keys[pygame.K_RIGHT]:
            slider_z += 0.05

        if dimensions == 2:
            latent = np.array([(mouse_x*max_x)/width+min_x, (mouse_y*max_y)/height+min_y]).reshape((1, 2))
        else:
            latent = np.array([(mouse_x*max_x)/width+min_x, (mouse_y*max_y)/height+min_y, slider_z]).reshape((1, dimensions))
        print(latent)
        prediction = np.round(decoder.predict(latent).reshape((28, 28)) * round_n) / round_n

        for i in range(28):
            for j in range(28):
                v = prediction[j, i] * 255
                pygame.draw.rect(screen, (v, v, v), pygame.Rect(w*i+0, h*j+0, w, h))
        # pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(10, 10, 20, 20))
        pygame.display.flip()
        clock.tick(40)