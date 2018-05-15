import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer


mod_dictionary = {
    '8qam_circular' : 0,
    'am-dsb' : 1,
    '8cpfsk' : 2,
    'lfm_squarewave' : 3,
    '8pam' : 4,
    'ofdm-64-bpsk' : 5,
    'lfm_sawtooth' : 6,
    '8gfsk': 7,
    '16qam': 8,
    'ofdm-16-bpsk': 9,
    '32qam_rect': 10,
    '4ask': 11,
    '16psk': 12,
    'am-ssb': 13,
    '2gfsk': 14,
    'ofdm-32-bpsk': 15,
    '2cpfsk' : 16,
    '4cpfsk' : 17,
    '64qam': 18,
    '4pam': 19,
    'ofdm-64-qpsk': 20,
    '4gfsk': 21,
    'ook': 22,
    '32qam_cross': 23,
    '8qam_cross': 24,
    'ofdm-32-qpsk': 25,
    'ofdm-16-qpsk': 26,
    'wbfm': 27,
    'bpsk': 28
}


def load_train_test_set(DF_LOAD_PATH, n_samples=10*1000):
    print("Loading data")
    # returns x_train, x_test in shape (60000, 64, 2, 1)

    ### new stuff

    df = pd.read_pickle(DF_LOAD_PATH)

    print("Dataframe shape:" + str(df.shape))

    df_sample = df.sample(n=n_samples).copy()
    df_sample.reset_index(drop=True)
    df_sample.reindex()

    data_as_array = df_sample.values

    QTscaler = QuantileTransformer()
    MMscaler = MinMaxScaler()

    MMscaler.fit(data_as_array[:, :256])
    data_as_array[:, :256] = MMscaler.transform(data_as_array[:, :256])

    QTscaler.fit(data_as_array[:, :256])
    data_as_array[:, :256] = QTscaler.transform(data_as_array[:, :256])

    split_index = int(len(data_as_array)*0.9)

    # 90:10 train:test split
    x_train = data_as_array[0:split_index, :]
    x_test = data_as_array[split_index:, :]

    print("Number of training samples: " + str(len(x_train)))
    print("Number of test samples: " + str(len(x_test)))



    # create verification set that contains one of each mod
    x_verification = np.empty(shape=[29, 258], dtype='O')
    for mod_index in range(len(x_verification)):
        test_index = 0
        for x in range(len(x_test)):
            if mod_dictionary[x_test[x,256]] == mod_index:
                test_index = x
                break

        x_verification[mod_index] = x_test[test_index,:]


    return x_train, x_test, x_verification


def view_data_set(FILE_PATH = "../data/mod_14_clean.pkl"):
    print("Loading data")
    # returns x_train, x_test in shape (60000, 64, 2, 1)

    f = open(FILE_PATH, "rb")
    mods, data = pickle.loads(f.read(), encoding='ISO-8859-1')

    print("loaded")



# load_train_test_set(DF_LOAD_PATH = "../data/mod_26_rsf")

"""
x_train, x_test, x_test_samples_by_mod  = load_train_test_set()

print(len(x_train))
print(len(x_test))

print(x_test_samples_by_mod)

print("done")
"""