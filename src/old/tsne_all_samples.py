import numpy as np
from sklearn.datasets import fetch_mldata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ggplot import *
import time
from sklearn.manifold import TSNE
import pickle

"""
mnist = fetch_mldata("MNIST original")
X = mnist.data / 255.0
y = mnist.target

print(X.shape,y.shape)
"""


### need to create X, y sets for modulation samples

FILE_PATH = "../data/mod_14_clean.pkl"
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

# flatten array of 64x2 images
X = all_mods_sorted.reshape((13*5000, 64*2))

# create an array of the modulation label
y = np.array([])
for mod in range(13):
    curr_mod = np.zeros(5000)
    curr_mod.fill(mod)
    y = np.append(y, curr_mod)


feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]


df = pd.DataFrame(X, columns=feat_cols)
df['label'] = y
df['label'] = df['label'].apply(lambda i: str(i))

X, y = None, None

print("Size of the dataframe: {}".format(df.shape))

rndperm = np.random.permutation(df.shape[0])


### PCA
"""
plt.gray()
fig = plt.figure(figsize=(16, 7))
for i in range(0, 30):
    ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']))

    ax.matshow(df.loc[rndperm[i], feat_cols].values.reshape((28, 28)).astype(float))
plt.show()
"""

"""
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1]
df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

chart = ggplot( df.loc[rndperm[:3000],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
#print(chart)
"""


#### BEGIN TSNE

n_sne = 10000
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by digit")
print(chart)



"""
#### PCA THEN TSNE
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)

print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

n_sne = 10000

time_start = time.time()

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50[rndperm[:n_sne]])

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne = None
df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne-pca'] = tsne_pca_results[:,0]
df_tsne['y-tsne-pca'] = tsne_pca_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by Digit (PCA)")
print(chart)
"""