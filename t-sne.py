import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from ggplot import ggplot, aes, geom_point, ggtitle
import plotly
import plotly.graph_objs as go


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
labels_df = pd.read_csv(os.path.join("F:/", "DL-code", "text2shape-data", "captions.csv"), usecols=['modelId', 'category'])
captions_dict = labels_df.to_dict(orient='records')
labels_dict = {}
for i in range(len(captions_dict)):
    if captions_dict[i]['modelId'] in model_id:
        labels_dict[captions_dict[i]['modelId']] = captions_dict[i]['category']
df['label'] = [labels_dict[id] for id in model_id]

print('Size of the dataframe: {}'.format(df.shape))

pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]
# df['pca_3'] = pca_result[:, 2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
df_pca = df.copy()
chart = ggplot(df_pca, aes(x = 'pca_1', y = 'pca_2', color='label')) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
chart.show()


time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_tsne = df.copy()
df_tsne['x_tsne'] = tsne_results[:, 0]
df_tsne['y_tsne'] = tsne_results[:, 1]
# df_tsne['z_tsne'] = tsne_results[:, 2]
chart = ggplot( df_tsne, aes(x = 'x_tsne', y = 'y_tsne', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
chart.show()


# embedding_var = tf.Variable(data, name=NAME_TO_VISUALISE_VARIABLE)
summary_writer = tf.summary.FileWriter(LOG_DIR)





