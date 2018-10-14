import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.datasets import mnist

def open_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

PATH = os.getcwd()
LOG_DIR = PATH + '/project-tensorboard/log-1/'


embeddings = open_pickle(os.path.join("F:/", "DL-code", "text2shape-data" ,"shapenet", "shapenet-embeddings", "text_embeddings_train.p"))
sample_tuple = embeddings['caption_embedding_tuples'][0]
embedding_shape = list(sample_tuple[3].shape)
assert len(embedding_shape) == 1

embedding_data = [item[3] for item in embeddings['caption_embedding_tuples']]
model_id = [item[2] for item in embeddings['caption_embedding_tuples']]
print("df shape: ", np.shape(embedding_data))
print("labels shape: ", np.shape(model_id))

feat_cols = ['col' + str(i) for i in range(np.shape(embedding_data)[1])]
df = pd.DataFrame(embedding_data, columns=feat_cols)
labels_df = pd.read_csv(os.path.join("F:/", "DL-code", "text2shape-data" ,"shapenet", "captions.csv"), usecols=['modelId', 'category'])
result = labels_df.to_dict(orient='records')
df['label'] = model_id

print('Size of the dataframe: {}'.format(df.shape))

pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]
df['pca_3'] = pca_result[:, 2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# embedding_var = tf.Variable(data, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)





