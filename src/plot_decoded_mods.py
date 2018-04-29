from matplotlib import pyplot as plt
import pickle

with open("../data/decoded_mods.pkl", 'rb') as f:
    decoded_imgs = pickle.load(f)
with open("../data/x_test_samples_by_mod.pkl", 'rb') as f:
    x_test = pickle.load(f)

FILE_PATH = "../data/mod_14_clean.pkl"
f = open(FILE_PATH, "rb")
mods, data = pickle.loads(f.read(), encoding='ISO-8859-1')
print(mods)

n = 13
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(64, 2))
    plt.title(mods[i], fontsize=9)
    plt.gray()
    plt.colorbar()
    #plt.show()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(64, 2))
    plt.gray()
    plt.colorbar()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #plt.show()

plt.show()

"""
# plotting images as their encodings:
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""