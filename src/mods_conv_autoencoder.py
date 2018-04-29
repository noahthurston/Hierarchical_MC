# keras convolutional autoencoder for MNIST
# Based on: https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import regularizers
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import datetime
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import pickle
import preprocessing

input_img = Input(shape=(64, 2, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (32, 1), activation='relu', padding='same', strides=1)(input_img)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Dense(128, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(4, activation='relu')(x)
encoded = Dense(2, activation='relu')(x)

print("Encoded tensor shape: " + str(encoded.get_shape().as_list()))

x = Dense(2, activation='relu')(encoded)
x = Dense(4, activation='relu')(x)
x = Dense(8, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = UpSampling2D((2, 1))(x)
x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
x = UpSampling2D((2, 1))(x)
x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
x = UpSampling2D((2, 1))(x)
x = Conv2D(64, (32, 1), activation='relu', padding='same')(x)

decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

print("Decoded tensor shape: " + str(decoded.get_shape().as_list()))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print(autoencoder.summary())

print("Model built")

x_train, x_test, x_test_samples_by_mod = preprocessing.load_train_test_set(FILE_PATH = "../data/mod_14_clean.pkl")


x_train = np.reshape(x_train, (len(x_train), 64, 2, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 64, 2, 1))  # adapt this if using `channels_first` image data format
x_test_samples_by_mod = np.reshape(x_test_samples_by_mod, (len(x_test_samples_by_mod), 64, 2, 1))

save_name = "cnn_autoencoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir=('../logs/' + save_name))])

decoded_mods = autoencoder.predict(x_test_samples_by_mod)

save_name = "cnn_autoencoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
save_path = "../data/" + save_name + ".h5"
print("saving to: " + save_path)
autoencoder.save(save_path)


encoder_model = Model(input_img, encoded)
encoder_model.compile(optimizer='adam', loss='mean_squared_error')
save_name = "cnn_encoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
save_path = "../data/" + save_name + ".h5"
print("saving to: " + save_path)
encoder_model.save(save_path)


with open("../data/decoded_mods.pkl", 'wb') as f:
    pickle.dump(decoded_mods, f, pickle.HIGHEST_PROTOCOL)
with open("../data/x_test_samples_by_mod.pkl", 'wb') as f:
    pickle.dump(x_test_samples_by_mod, f, pickle.HIGHEST_PROTOCOL)

