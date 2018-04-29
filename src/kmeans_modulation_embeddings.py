import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def kmeans_embedded_modulation_samples(DF_PATH):
    # load dataframe
    df = pd.read_pickle(DF_PATH)
    print("Size of the dataframe: {}".format(df.shape))
    rndperm = np.random.permutation(df.shape[0])


    feat_cols = df.columns.values[:-1]

    n_sne = 10000
    n_clusters = 2


    km = KMeans(n_clusters=n_clusters)
    km_results = km.fit(df.loc[rndperm[:n_sne], feat_cols].values)

    df_kmeans = df.loc[rndperm[:n_sne], :].copy()

    cluster_results = km.labels_

    df_kmeans['cluster'] = cluster_results

    mods = np.array(df_kmeans.loc[:, 'label'])
    mods = mods.astype(np.float).astype(np.int)

    clusters = np.array(df_kmeans.loc[:, 'cluster'])
    clusters.astype(np.int)

    bin_counts = np.bincount(mods)

    print(mods)
    print(clusters)
    print(bin_counts)

    normal_composition_matrix = np.zeros((n_clusters,13), dtype=np.int)

    for index, cluster in enumerate(clusters):
        normal_composition_matrix[cluster, mods[index]] += 1

    print(normal_composition_matrix)

    percent_composition_matrix = np.zeros((n_clusters, 13), dtype=np.float32)

    for cluster, cluster_arr in enumerate(normal_composition_matrix):
        for mod_index, mod_freq in enumerate(cluster_arr):
            percent_composition_matrix[cluster,mod_index] = normal_composition_matrix[cluster, mod_index]/bin_counts[mod_index]

    print(percent_composition_matrix)



kmeans_embedded_modulation_samples(DF_PATH = "../data/encoded_vectors_df_04-29--02-11")