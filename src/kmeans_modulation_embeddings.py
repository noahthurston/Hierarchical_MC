import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

def kmeans_embedded_modulation_samples(DF_PATH):
    # load dataframe
    df = pd.read_pickle(DF_PATH)
    print("Size of the dataframe: {}".format(df.shape))


    # sort dataframe to only cluster certan modulations
    all_mods = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0', '12.0']
    included_mods = ['0.0', '2.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0']
    num_mods = len(included_mods)

    mask = df['label'].isin(included_mods)

    pick1 = (df['label'] == '0.0')
    pick2 = (df['label'] == '1.0')
    pick3 = (df['label'] == '3.0')
    #df_sorted.replace([np.inf, -np.inf], np.nan).dropna(how='all')


    df_sorted = df[mask].reset_index(drop=True)


    feat_cols = df_sorted.columns.values[:-1]

    n_kmeans = 10000
    n_clusters = 3

    rndperm = np.random.permutation(df_sorted.shape[0])


    km = KMeans(n_clusters=n_clusters)
    km_results = km.fit( df_sorted.loc[rndperm[:n_kmeans], feat_cols].values)

    df_kmeans = df_sorted.loc[rndperm[:n_kmeans], :].copy()

    cluster_results = km.labels_

    df_kmeans['cluster'] = cluster_results

    mods = np.array(df_kmeans.loc[:, 'label'])
    mods = mods.astype(np.float).astype(np.int)
    #mods = range(num_mods)

    clusters = np.array(df_kmeans.loc[:, 'cluster'])
    clusters.astype(np.int)

    bin_counts = np.zeros((13))
    bin_counts_tmp = np.append(np.bincount(mods), np.zeros(1))
    for x in range(len(bin_counts_tmp)): bin_counts[x] = bin_counts_tmp[x]


    print(mods)
    print(clusters)
    print(bin_counts)

    normal_composition_matrix = np.zeros((n_clusters, 13), dtype=np.int)

    for index, cluster in enumerate(clusters):
        normal_composition_matrix[cluster, mods[index]] += 1

    print(normal_composition_matrix)

    percent_composition_matrix = np.zeros((n_clusters, 13), dtype=np.float32)

    for cluster, cluster_arr in enumerate(normal_composition_matrix):
        for mod_index, mod_freq in enumerate(cluster_arr):
            percent_composition_matrix[cluster, mod_index] = normal_composition_matrix[cluster, mod_index]/bin_counts[mod_index]

    percent_composition_matrix[np.isnan(percent_composition_matrix)] = 0

    print(percent_composition_matrix)


kmeans_embedded_modulation_samples(DF_PATH = "../data/encoded_vectors_df_04-29--02-11.pkl")