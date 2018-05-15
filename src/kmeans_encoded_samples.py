import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

def kmeans_encoded_samples(DF_PATH):
    # load dataframe
    df = pd.read_pickle(DF_PATH)
    print("Size of the dataframe: {}".format(df.shape))

    all_mod_labels = [
    '8qam_circular',
    'am-dsb',
    '8cpfsk',
    'lfm_squarewave',
    '8pam',
    'ofdm-64-bpsk',
    'lfm_sawtooth',
    '8gfsk',
    '16qam',
    'ofdm-16-bpsk',
    '32qam_rect',
    '4ask',
    '16psk',
    'am-ssb',
    '2gfsk',
    'ofdm-32-bpsk',
    '2cpfsk',
    '4cpfsk',
    '64qam',
    '4pam',
    'ofdm-64-qpsk',
    '4gfsk',
    'ook',
    '32qam_cross',
    '8qam_cross',
    'ofdm-32-qpsk',
    'ofdm-16-qpsk',
    'wbfm',
    'bpsk'
    ]

    included_mod_labels = [
    'am-dsb',
    '8cpfsk',
    'lfm_squarewave',
    '8pam',
    'lfm_sawtooth',
    'ofdm-16-bpsk',
    'am-ssb',
    '2gfsk',
    'ofdm-32-bpsk',
    '2cpfsk',
    '4cpfsk',
    '4pam',
    'ook',
    'ofdm-32-qpsk',
    'ofdm-16-qpsk',
    'wbfm',
    ]

    # macro variables
    N_KMEANS_SAMPLES = 100*1000
    N_CLUSTERS = 2
    N_MODS = 29

    num_mods = len(included_mod_labels)

    # create mask to only k-means subset of mods
    mask = df['mod_name'].isin(included_mod_labels)
    df_sorted = df[mask].reset_index(drop=True)
    #df_sorted = df

    feat_cols = df_sorted.columns.values[:-4]

    # create permutation for random sample
    rndperm = np.random.permutation(df_sorted.shape[0])

    # create sklearn kmeans object
    km = KMeans(n_clusters=N_CLUSTERS)
    km_results = km.fit(df_sorted.loc[rndperm[:N_KMEANS_SAMPLES], feat_cols].values)

    # create dataframe for kmeans results
    df_kmeans = df_sorted.loc[rndperm[:N_KMEANS_SAMPLES], :].copy()
    cluster_results = km.labels_
    df_kmeans['cluster'] = cluster_results

    mods_of_samples = np.array(df_kmeans.loc[:, 'mod_name'])
    #mods_of_samples = mods_of_samples.astype(np.float).astype(np.int)
    mods_of_samples = [mod_dictionary[x] for x in mods_of_samples]

    clusters_of_samples = np.array(df_kmeans.loc[:, 'cluster'])
    clusters_of_samples.astype(np.int)

    # count how many of each mod are in each cluster
    bin_counts = np.zeros((N_MODS))
    for index, count in enumerate(np.bincount(mods_of_samples)): bin_counts[index] = count

    print(mods_of_samples)
    print(clusters_of_samples)
    print(bin_counts)

    normal_composition_matrix = np.zeros((N_CLUSTERS, N_MODS), dtype=np.int)

    for index, cluster in enumerate(clusters_of_samples):
        normal_composition_matrix[cluster, mods_of_samples[index]] += 1

    print(normal_composition_matrix)

    percent_composition_matrix = np.zeros((N_CLUSTERS, N_MODS), dtype=np.float32)

    # create matrix to show what percent of each modulation is in each cluster
    for cluster, cluster_arr in enumerate(normal_composition_matrix):
        for mod_index, mod_freq in enumerate(cluster_arr):
            percent_composition_matrix[cluster, mod_index] = normal_composition_matrix[cluster, mod_index]/bin_counts[mod_index]

    percent_composition_matrix[np.isnan(percent_composition_matrix)] = 0

    print(percent_composition_matrix)
    print((percent_composition_matrix*1.99).astype(int))

def create_hierarchical_df(mod_hierarchy_dict, DF_LOAD_PATH, DF_SAVE_PATH):
    # load dataframe
    df = pd.read_pickle(DF_LOAD_PATH)
    print("Size of the dataframe: {}".format(df.shape))

    #df.apply(lambda row: mod_hierarchy_dict[int(float(row['label']))], axis=1)
    df['hier'] = df['mod_name'].apply(lambda row: mod_hierarchy_dict[row])

    df.to_pickle(DF_SAVE_PATH)

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

# first heirarchy
#[[0 1 1 1 0 1 0 0 1 0 0 0 1 1 1 0 1 0 0 1 0 0 1 1 1 0]
# [1 0 0 0 1 0 1 1 0 1 1 1 0 0 0 1 0 1 1 0 1 1 0 0 0 1]]
"""
mod_hierarchy_dict = {
    0: (0),
    1: (0,0,0),
    2: (0,0,1),
    3: (0,0,2),
    4: (0),
    5: (0,0,3),
    6: (0),
    7: (0),
    8: (0,0,4),
    9: (0),
    10: (0),
    11: (0),
    12: (0,0,5),
    13: (0,0,6),
    14: (0,0,7),
    15: (0),
    16: (1),
    17: (0),
    18: (0),
    19: (1),
    20: (0),
    21: (0),
    22: (1),
    23: (1),
    24: (1),
    25: (0),
}
"""

mod_hierarchy_dict = {
    'am-dsb' : (0,0,0),
    'lfm_squarewave' : (0,0,1),
    '8pam' : (0,0,2),
    'lfm_sawtooth' : (0,0,3),
    'ofdm-16-bpsk' : (0,0,4),
    'am-ssb' : (0,0,5),
    '4pam' : (0,0,6),
    'ook' : (0,0,7),
    'ofdm-16-qpsk' : (0, 0, 8),
    '2gfsk' : (0,1,0),
    'ofdm-32-bpsk' : (0,1,1),
    'ofdm-32-qpsk' : (0,1,2),
    'wbfm' : (0,1,3),
    '8qam_circular' : (1,0,0),
    'ofdm-64-bpsk' : (1,0,1),
    '8gfsk' : (1,0,2),
    '16qam'  : (1,0,3),
    '32qam_rect'  : (1,0,4),
    '16psk'  : (1,0,5),
    '64qam'  : (1,0,6),
    'ofdm-64-qpsk'  : (1,0,7),
    '32qam_cross'  : (1,0,8),
    '8qam_cross' : (1,0,9),
    'bpsk'  : (1,0,10),
    '4ask'  : (1,1,0),
    '4gfsk'  : (1,0,1)
}


#create_hierarchical_df(mod_hierarchy_dict, DF_LOAD_PATH="../data/mod_29_rsf", DF_SAVE_PATH="../data/encoded_128vectors_df_05-14--23-59.pkl")
kmeans_encoded_samples(DF_PATH="../data/encoded_128vectors_df_05-14--23-59.pkl")


"""
cluster 0
    cluster 00
        '8qam_circular',
        '8gfsk',
        '16qam',
        '32qam_rect',
        '4ask',
        '16psk',
        '64qam',
        '4gfsk',
        '32qam_cross',
        '8qam_cross',
        'bpsk'
    
    cluster 01
        'ofdm-64-qpsk',
        'ofdm-64-bpsk',
    
    
cluster 1
    cluster 10
        'am-dsb',
        '8cpfsk',
        'lfm_squarewave',
        '8pam',
        'lfm_sawtooth',
        'ofdm-16-bpsk',
        'am-ssb',
        '2gfsk',
        '2cpfsk',
        '4cpfsk',
        '4pam',
        'ook',
        'ofdm-16-qpsk',
        'wbfm',

    cluster 11
        'ofdm-32-bpsk',
        'ofdm-32-qpsk',
    

"""