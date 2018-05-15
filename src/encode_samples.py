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
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer



### need to create X, y sets for modulation samples

"""
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
"""


N_SAMPLES =520*1000
DF_LOAD_PATH="../data/mod_26_rsf"
ENCODER_FILE_PATH='../models/cnn_encoder_05-06--10-44.h5'
ENCODED_LENGTH = 256

df_samples = pd.read_pickle(DF_LOAD_PATH)

data_as_array = df_samples.values

QTscaler = QuantileTransformer()
MMscaler = MinMaxScaler()

MMscaler.fit(data_as_array[:, :256])
data_as_array[:, :256] = MMscaler.transform(data_as_array[:, :256])

QTscaler.fit(data_as_array[:, :256])
data_as_array[:, :256] = QTscaler.transform(data_as_array[:, :256])

# load encoder
encoder = load_model(ENCODER_FILE_PATH)

#encoder.compile(optimizer='adam', loss='mean_squared_error')
encoded_samples = encoder.predict(data_as_array[:,0:256].astype(float).reshape( N_SAMPLES, 128, 2, 1))

encoded_samples = encoded_samples.reshape(-1, ENCODED_LENGTH)

encoded_column_labels = ['pixel'+str(i) for i in range(ENCODED_LENGTH)]
label_columns_labels = df_samples.columns.values[-4:]

df_encoded_columns = pd.DataFrame(encoded_samples, columns=encoded_column_labels)
df_labels_columns = pd.DataFrame(df_samples[label_columns_labels].values, columns=label_columns_labels)

df_encoded = pd.concat([df_encoded_columns,df_labels_columns], axis=1)

DF_PATH = '../data/encoded_128vectors_df_' + datetime.datetime.now().strftime("%m-%d--%H-%M") + ".pkl"
print("saving dataframe to: " + DF_PATH)
df_encoded.to_pickle(DF_PATH)
