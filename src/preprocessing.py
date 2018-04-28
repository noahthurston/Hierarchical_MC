import numpy as np
import pickle
from matplotlib import pyplot as plt


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

def load_train_test_set(FILE_PATH = "../data/mod_14_clean.pkl"):
    print("Loading data")
    # returns x_train, x_test in shape (60000, 64, 2, 1)

    f = open(FILE_PATH, "rb")
    mods, data = pickle.loads(f.read(), encoding='ISO-8859-1')

    all_mods_separate = np.zeros((13,5000,64,2))
    # print(all_mods_separate)

    for mod_index in range(13):
        all_mods_separate[mod_index] = data[mods[mod_index]]

    # print(all_mods_separate)

        all_mods_together = all_mods_separate.reshape((13*5000, 64,2))

    # total 13*5k=65k samples
    # x_train: 50k samples
    # x_test: 15k samples

    x_train = (all_mods_together[0:50000]+1)/2
    x_test = (all_mods_together[50000:65000]+1)/2

    #### TO DO: should normalize data to be [0,1]

    x_test_samples_by_mod = np.zeros((13,64,2))
    random_sample_indexes = np.random.random_integers(0,5000,13)
    for index in range(13):
        x_test_samples_by_mod[index] = all_mods_separate[index][random_sample_indexes[index]]

    #raise SystemExit
    return x_train, x_test, x_test_samples_by_mod


x_train, x_test, x_test_samples_by_mod  = load_train_test_set()

print(len(x_train))
print(len(x_test))

print(x_test_samples_by_mod)

print("done")