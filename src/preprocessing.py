import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer


"""
mods (13 in total):
0: ook
1: bpsk
2: qpsk
3: 4ask
4: 4pam
5: 8psk
6: 8pam
7: 8qam_cross
8: 8qam_circular
9: 16qam
10: 16psk
11: 32qam_cross
12: 32qam_rect64qam

-modulation
    -5k length array
        -64 array length
            -2 array length
"""

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
    x_verification = np.empty(shape=[26, 260], dtype='O')
    for mod_index in range(len(x_verification)):
        x_verification[mod_index] = x_test[x_test[:, 257] == mod_index][0]

    print("debug")

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