#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:53:15 2018
Description:
"""
import numpy as np
import keras
#import urllib2
import pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.models import load_model, Model
from sklearn.model_selection import train_test_split
#from classify.mc_dataset import load_data # necessary to gen aribtrary data

from keras.layers import Input
from keras.callbacks import TensorBoard
import datetime

BATCH_SIZE = 128
EPOCHS = 100
NUM_FEATURES = 128
DATA_URL = "https://s3.amazonaws.com/radio-machine-learning/mod_14_clean.pkl"
FILE_PATH = "../data/mod_14_clean.pkl"

def train():
    # %% Get the data and prepare it as necessary
    x = []
    lbl = []

    f = open(FILE_PATH, "rb")
    #mods, data = pickle.loads(f.read())
    mods, data = pickle.loads(f.read(), encoding='ISO-8859-1')

    # Go through the list of modulations and create numeric labels
    # this is currently done in a way to support a unique label
    # per SNR or multi-labelling later
    for mod in mods:
        x.append(data[mod])
        for i in range(data[mod].shape[0]):
            lbl.append(mod)

    x = np.vstack(x) # stack it up my friend
    x_train, x_test, y_train, y_test = train_test_split(x, lbl, test_size=0.33, random_state=42)


    y_train = keras.utils.to_categorical(list(map(lambda x: mods.index(y_train[x]), range(len(y_train)))))
    y_test = keras.utils.to_categorical(list(map(lambda x: mods.index(y_test[x]), range(len(y_test)))))

    # %% Make the model

    """
    in_shp = list(x_train.shape[1:])
    model = Sequential()
    model.add(Conv1D(filters=128,
                     kernel_size=2,
                     strides=1,
                     padding='valid',
                     activation="relu",
                     name="conv1",
                     kernel_initializer='glorot_uniform',
                     input_shape=in_shp))
    model.add(Dropout(.5))
    model.add(MaxPooling1D(pool_size=1,padding='valid', name="pool1"))
    model.add(Conv1D(filters=128,
                     kernel_size=4,
                     strides=4,
                     padding='valid',
                     activation="relu",
                     name="conv2",
                     kernel_initializer='glorot_uniform',
                     input_shape=in_shp))
    model.add(Dropout(.5))
    model.add(MaxPooling1D(pool_size=1, padding='valid', name="pool2"))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"))
    model.add(Dropout(.5))
    model.add(Dense(len(mods), kernel_initializer='he_normal', name="dense2" ))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    """

    # begin build
    input_img = Input(shape=(64, 2, 1))  # adapt this if using `channels_first` image data format
    #input_img = list(x_train.shape[1:])

    x = Conv2D(128, (32, 1), activation='relu', padding='same', strides=1)(input_img)
    x = MaxPooling2D((2, 1), padding='same')(x)
    x = Conv2D(64, (16, 2), activation='relu', padding='same', strides=1)(x)
    x = MaxPooling2D((2, 1), padding='same')(x)
    x = Conv2D(32, (16, 2), activation='relu', padding='same', strides=1)(x)
    x = MaxPooling2D((2, 1), padding='same')(x)
    # x = MaxPooling2D((2, 1), padding='same')(x)
    # encoded = Conv2D(16, (1, 1), activation='relu', padding='same', strides=1)(x)

    x = Dense(128, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(64, activation='relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(32, activation='relu', kernel_initializer='he_normal')(x)

    x = Dense(13, kernel_initializer='he_normal', name="dense2", activation='softmax')(x)

    classifier = Model(input_img, x)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print(classifier.summary())

    """
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

    # x = Conv2D(4, (1, 1), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 1))(x)

    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)
    # end build
    """

    print(classifier.summary())

    save_name = "cnn_classifier_" + datetime.datetime.now().strftime("%m-%d--%H-%M")

    classifier.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks = [TensorBoard(log_dir=('../logs/' + save_name))])


    save_path = "../data/" + save_name + ".h5"
    print("saving to: " + save_path)
    classifier.save(save_path)

    # TODO: It would be nice to pickle up the model here just in case

def evaulate(load_path):
    # %% Get the data and prepare it as necessary
    x = []
    lbl = []

    f = open(FILE_PATH, "rb")
    #mods, data = pickle.loads(f.read())
    mods, data = pickle.loads(f.read(), encoding='ISO-8859-1')

    # Go through the list of modulations and create numeric labels
    # this is currently done in a way to support a unique label
    # per SNR or multi-labelling later
    for mod in mods:
        x.append(data[mod])
        for i in range(data[mod].shape[0]):
            lbl.append(mod)

    x = np.vstack(x) # stack it up my friend
    x_train, x_test, y_train, y_test = train_test_split(x, lbl, test_size=0.33, random_state=42)


    y_train = keras.utils.to_categorical(list(map(lambda x: mods.index(y_train[x]), range(len(y_train)))))
    y_test = keras.utils.to_categorical(list(map(lambda x: mods.index(y_test[x]), range(len(y_test)))))



    model = load_model(load_path)

    # %% Visualize the results. Can haz confusion matrix?
    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


train()
#evaulate("cnn_classifier_04-27--18-46.h5")