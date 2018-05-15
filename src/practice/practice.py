import numpy as np
import pickle
import pandas as pd

DF_LOAD_PATH = "../../data/mod_26_rsf_with_hier.pkl"
df = pd.read_pickle(DF_LOAD_PATH)
df_head = df.head()

arr = df.values

print("done")

raise SystemExit


N_SAMPLES = 10
DF_LOAD_PATH="../../data/mod_26_rsf"
ENCODED_LENGTH = 4

df = pd.read_pickle(DF_LOAD_PATH)

df_sample = df.sample(n=N_SAMPLES).copy()
df_sample.reset_index(drop=True)
df_sample.reindex()


encoded_samples = np.arange(0,N_SAMPLES*ENCODED_LENGTH)
encoded_samples = encoded_samples.reshape((N_SAMPLES,ENCODED_LENGTH))

encoded_column_labels = ['pixel'+str(i) for i in range(ENCODED_LENGTH)]
label_columns_labels = df.columns.values[-4:]

df_encoded_columns = pd.DataFrame(encoded_samples, columns=encoded_column_labels)
df_labels_columns = pd.DataFrame(df_sample[label_columns_labels].values, columns=label_columns_labels)

df_encoded = pd.concat([df_encoded_columns,df_labels_columns], axis=1)

print("done")