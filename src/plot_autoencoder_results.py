from matplotlib import pyplot as plt
import pickle
import datetime

with open("../data/verification_set_05-06--10-44.pkl", 'rb') as f:
    verification_set = pickle.load(f)

x_in = verification_set[0]
x_out = verification_set[1]
x_labels = verification_set[2]

n = 26
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_in[i].reshape(16, 16), clim=(0.0, 1.0))
    plt.title(x_labels[i][-4], fontsize=9)
    plt.gray()
    plt.colorbar()
    #plt.show()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(x_out[i].reshape(16, 16), clim=(0.0, 1.0))
    plt.gray()
    plt.colorbar()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #plt.show()

plt.savefig("../data/autoencoder_comparison_" + datetime.datetime.now().strftime("%m-%d--%H-%M") + ".png")

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