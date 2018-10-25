import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import utils
from tensorflow.contrib.tensorboard.plugins import projector
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

thumb_dir = os.path.join(MODEL_DIR,"resize_models")
image_data = utils.get_images(thumb_dir)


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

utils.write_metadata(os.path.join(thumb_dir, "metadata.tsv"), df['label'])

with tf.Session() as sess:
    # assign the tensor that we want to visualize to the embedding variable
    embedding_var = tf.Variable(np.shape(embedding_data), name="embedding")
    sess.run(embedding_var.initializer)
    init = tf.global_variables_initializer()
    init.run()
    config = projector.ProjectorConfig()
    config.model_checkpoint_path = os.path.join(PATH, "tsne", 'my-model.ckpt')
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = os.path.join(os.path.join(thumb_dir, "metadata.tsv"))
    embedding.sprite.image_path = os.path.join(thumb_dir, "sprite.jpg")
    embedding.sprite.single_image_dim.extend([utils.img_w, utils.img_h])
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    projector.visualize_embeddings(summary_writer, config)
    saver_embed = tf.train.Saver([embedding_var])
    saver_embed.save(sess, os.path.join(PATH, "tsne", 'my-model.ckpt'))
    sess.run(embedding_var)






