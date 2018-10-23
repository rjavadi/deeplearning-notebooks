import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import utils
from ggplot import ggplot, aes, geom_point, ggtitle


PATH = os.getcwd()
LOG_DIR = PATH + '/project-tensorboard/log-1/'
MODEL_DIR = os.path.join("F:/", "DL-code" ,"text2shape-data", "nrrd_256_filter_div_32_solid")


embeddings = utils.open_pickle(os.path.join("F:/", "DL-code", "text2shape-data" ,"shapenet", "shapenet-embeddings", "text_embeddings_train.p"))
sample_tuple = embeddings['caption_embedding_tuples'][0]
embedding_shape = list(sample_tuple[3].shape)
assert len(embedding_shape) == 1

embedding_data = [item[3] for item in embeddings['caption_embedding_tuples']]
model_id = [item[2] for item in embeddings['caption_embedding_tuples']]
print("embedding shape: ", np.shape(embedding_data))
print("model_id shape: ", np.shape(model_id))

# utils.resize_images(MODEL_DIR, model_id[0])
thumb_dir = os.path.join(MODEL_DIR,"resize_models")
image_data = utils.get_images(thumb_dir)
# utils.write_sprite_image(os.path.join(thumb_dir, "sprite.png"), image_data)

sprite = utils.images_to_sprite(np.array(image_data), os.path.join(thumb_dir, "sprite.jpg"))


feat_cols = ['col' + str(i) for i in range(np.shape(embedding_data)[1])]
df = pd.DataFrame(embedding_data, columns=feat_cols)



"""Reading categories to use it as a label for coloring/separating data on chart"""
labels_df = pd.read_csv(os.path.join("F:/", "DL-code", "text2shape-data", "captions.csv"), usecols=['modelId', 'category'])

""" reformat it as a list of dicts """
captions_dict = labels_df.to_dict(orient='records')

""" making a dictionary with format: #model_id: Category """
labels_dict = {}
for i in range(len(captions_dict)):
    if captions_dict[i]['modelId'] in model_id:
        labels_dict[captions_dict[i]['modelId']] = captions_dict[i]['category']


# the order of labels and models is preserved
df['label'] = [labels_dict[id] for id in model_id]

print('Size of the dataframe: {}'.format(df.shape))


# Apply PCA transform
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


tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)

df_tsne = df.copy()
df_tsne['x_tsne'] = tsne_results[:, 0]
df_tsne['y_tsne'] = tsne_results[:, 1]
# df_tsne['z_tsne'] = tsne_results[:, 2]
chart = ggplot( df_tsne, aes(x = 'x_tsne', y = 'y_tsne', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("First and Second Principal Components colored by digit")
chart.show()


summary_writer = tf.summary.FileWriter(LOG_DIR)





