# uses the first half the trained autoencoder to encode the modulation samples as embedded vectors

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import regularizers
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import datetime
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
import pickle
import preprocessing
import pandas as pd


### need to create X, y sets for modulation samples

FILE_PATH = "../data/mod_14_clean.pkl"
ENCODED_VECTOR_LENGTH = 128

f = open(FILE_PATH, "rb")
mods, data = pickle.loads(f.read(), encoding='ISO-8859-1')

# convert from dictionary to np.array
all_mods_sorted = np.zeros((13, 5000, 64, 2))
for mod_index in range(13):
        all_mods_sorted[mod_index] = data[mods[mod_index]]

# normalize data to be between 0 and 1
min_val = np.min(all_mods_sorted)
all_mods_sorted = all_mods_sorted + np.abs(min_val)
max_val = np.max(all_mods_sorted)
all_mods_sorted = all_mods_sorted / max_val


SAMP_SIZE = 13*5000

# reshaped so they can be fed into encoder
all_unencoded_samples = all_mods_sorted.reshape((13*5000, 64, 2, 1))
unencoded_samples = all_unencoded_samples[:SAMP_SIZE]


# load encoder
ENCODER_FILE_PATH='../models/cnn_encoder_04-30--22-59.h5'
encoder = load_model(ENCODER_FILE_PATH)

#encoder.compile(optimizer='adam', loss='mean_squared_error')
encoded_samples = encoder.predict(unencoded_samples)

encoded_samples = encoded_samples.reshape(SAMP_SIZE, ENCODED_VECTOR_LENGTH)



# create an array of the modulation label
mod_labels = np.array([])
for mod in range(13):
    curr_mod = np.zeros(5000)
    curr_mod.fill(mod)
    mod_labels = np.append(mod_labels, curr_mod)

feat_cols = ['pixel'+str(i) for i in range(encoded_samples.shape[1])]

df = pd.DataFrame(encoded_samples, columns=feat_cols)
df['label'] = mod_labels
df['label'] = df['label'].apply(lambda i: str(i))

encoded_samples, mod_labels = None, None

print("Size of the dataframe: {}".format(df.shape))

rndperm = np.random.permutation(df.shape[0])

DF_PATH = '../data/encoded_128vectors_df_' + datetime.datetime.now().strftime("%m-%d--%H-%M") + ".pkl"
print("saving dataframe to: " + DF_PATH)
df.to_pickle(DF_PATH)
