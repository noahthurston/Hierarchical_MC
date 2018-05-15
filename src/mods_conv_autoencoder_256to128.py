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

input_img = Input(shape=(128, 2, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (32, 1), activation='relu', padding='same', strides=1)(input_img)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Conv2D(32, (8, 2), activation='relu', padding='same', strides=1)(x)
x = MaxPooling2D((2, 1), padding='same')(x)
x = Dense(128, activation='relu')(x)
x = Dense(32, activation='relu')(x)
#x = Dense(8, activation='relu')(x)
encoded = Dense(8, activation='relu')(x)

print("Encoded tensor shape: " + str(encoded.get_shape().as_list()))

x = Dense(8, activation='relu')(encoded)

#x = Dense(8, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = UpSampling2D((2, 1))(x)
x = Conv2D(32, (8, 2), activation='relu', padding='same', strides=1)(x)
x = UpSampling2D((2, 1))(x)
x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
x = UpSampling2D((2, 1))(x)
x = Conv2D(32, (32, 1), activation='relu', padding='same')(x)

decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

print("Decoded tensor shape: " + str(decoded.get_shape().as_list()))

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print(autoencoder.summary())

print("Model built")


N_SAMPLES =100*1000
x_train, x_test, x_verification = preprocessing.load_train_test_set(DF_LOAD_PATH="../data/mod_26_rsf", n_samples=N_SAMPLES)

x_train = np.reshape(x_train[:,0:256].astype(float), (len(x_train), 128, 2, 1))
x_test = np.reshape(x_test[:,0:256].astype(float), (len(x_test), 128, 2, 1))
x_verification_in = np.reshape(x_verification[:,0:256].astype(float), (len(x_verification), 128, 2, 1))

save_name = "cnn_autoencoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir=('../logs/' + save_name))])

x_verification_out = autoencoder.predict(x_verification_in)

save_name = "cnn_autoencoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
save_path = "../models/" + save_name + ".h5"
print("saving to: " + save_path)
autoencoder.save(save_path)

encoder_model = Model(input_img, encoded)
encoder_model.compile(optimizer='adam', loss='mean_squared_error')
save_name = "cnn_encoder_" + datetime.datetime.now().strftime("%m-%d--%H-%M")
save_path = "../models/" + save_name + ".h5"
print("saving to: " + save_path)
encoder_model.save(save_path)

verification_set = (x_verification_in, x_verification_out, x_verification[:,-4:])

with open("../data/verification_set_" + datetime.datetime.now().strftime("%m-%d--%H-%M") + ".pkl", 'wb') as f:
    pickle.dump(verification_set, f, pickle.HIGHEST_PROTOCOL)
