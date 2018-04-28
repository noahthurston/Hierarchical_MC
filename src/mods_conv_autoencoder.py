# keras convolutional autoencoder for MNIST
# Based on: https://blog.keras.io/building-autoencoders-in-keras.html

# first imports
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.datasets import mnist
import numpy as np
import datetime

from keras.callbacks import TensorBoard

from matplotlib import pyplot as plt

import pickle


import preprocessing

# images need to be 64x2

input_img = Input(shape=(64, 2, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (32, 1), activation='relu', padding='same', strides=1)(input_img)
#x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(16, (16, 2), activation='relu', padding='same', strides=1)(x)
#x = MaxPooling2D((2, 1), padding='same')(x)
#encoded = Conv2D(16, (1, 1), activation='relu', padding='same', strides=1)(x)

x = Dense(128, activation='relu')(x)
encoded = Dense(64, activation='relu')(x)

print("Encoded tensor shape: " + str(encoded.get_shape().as_list()))

x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
x = Conv2D(16, (16, 2), activation='relu', padding='same', strides=1)(x)
x = Conv2D(31, (32, 1), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 1))(x)
#x = Conv2D(4, (1, 1), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 1))(x)

decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)


print("Decoded tensor shape: " + str(decoded.get_shape().as_list()))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print(autoencoder.summary())

print("Model built")

#raise SystemExit

# adapt to create train/test sets

# To train it, we will use the original MNIST digits with shape (samples, 3, 28, 28),
# and we will just normalize pixel values between 0 and 1.
#(x_train, _), (x_test, _) = preprocessing.load_data()


#### TO DO: should normalize data to be [0,1]
x_train, x_test, x_test_samples_by_mod = preprocessing.load_train_test_set(FILE_PATH = "../data/mod_14_clean.pkl")



"""
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
"""

x_train = np.reshape(x_train, (len(x_train), 64, 2, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 64, 2, 1))  # adapt this if using `channels_first` image data format
x_test_samples_by_mod = np.reshape(x_test_samples_by_mod, (len(x_test_samples_by_mod), 64, 2, 1))

save_name = "cnn_autoencoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

# tensorboard backend
# tensorboard --logdir=/tmp/autoencoder
#autoencoder.fit(x_train, x_train,epochs=1,batch_size=128,shuffle=True,validation_data=(x_test, x_test),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir=('../logs/' + save_name))])

# Let's take a look at the reconstructed digits:
decoded_imgs = autoencoder.predict(x_test_samples_by_mod)

with open("decoded_imgs.pkl", 'wb') as f:
    pickle.dump(decoded_imgs, f, pickle.HIGHEST_PROTOCOL)
with open("x_test_samples_by_mod.pkl", 'wb') as f:
    pickle.dump(x_test_samples_by_mod, f, pickle.HIGHEST_PROTOCOL)


