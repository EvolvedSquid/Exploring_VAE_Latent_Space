from keras.models import load_model
from keras.layers import Layer
import keras.backend as K
import numpy as np
import pygame


def nll(y_true, y_pred): # Negative loss likelihood
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

class KLDivergenceLayer(Layer): # KL Layer
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

encoder = load_model('Models/encoder_3d.hdf5', custom_objects={'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})
decoder = load_model('Models/decoder_3d.hdf5', custom_objects={'KLDivergenceLayer': KLDivergenceLayer, 'nll': nll})

round_n = 3 # Round visuals to nearest 1/3
dimensions = 3 # Number of dimensions
width = height = 1008
w = h = 36
min_x = -2 # Min x value possible with mouse
max_x = 3 # Max x value possible with mouse
min_y = -2 # Min y value possible with mouse
max_y = 3 # Max y value possible with mouse


slider_z = 0.5
slider_t = 0.5

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
        if keys[pygame.K_UP]:
            slider_t += 0.05
        if keys[pygame.K_DOWN]:
            slider_t -= 0.05
        # Get latent dimensions from mouse position and value of slider_z/t
        if dimensions == 2:
            latent = np.array([(mouse_x*max_x)/width+min_x, (mouse_y*max_y)/height+min_y]).reshape((1, 2))
        else:
            latent = np.array([(mouse_x*max_x)/width+min_x, (mouse_y*max_y)/height+min_y, slider_z]).reshape((1, dimensions))
        print(latent)
        prediction = np.round(decoder.predict(latent).reshape((28, 28)) * round_n) / round_n

        # Draw character
        for i in range(28):
            for j in range(28):
                v = prediction[j, i] * 255
                pygame.draw.rect(screen, (v, v, v), pygame.Rect(w*i+0, h*j+0, w, h))
        pygame.display.flip()
        clock.tick(40)
