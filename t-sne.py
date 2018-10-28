import os
import tensorflow as tf
import numpy as np
import pandas as pd
import utils
from tensorflow.contrib.tensorboard.plugins import projector


PATH = os.getcwd()
LOG_DIR = PATH + '/project-tensorboard/log-1'
MODEL_DIR = os.path.join("F:/", "DL-code" ,"text2shape-data", "nrrd_256_filter_div_32_solid")
# METADATA_DIR = os.path.join('project-tensorboard', 'log-1', 'metadata.tsv')
# SPRITE_DIR = os.path.join('project-tensorboard', 'log-1', 'sprite.jpg')


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

# TODO: rewrite images and create sprites again. compare your code with github sample of t-sne.

# the order of labels and models is preserved
df['label'] = [labels_dict[id] for id in model_id]

print('Size of the dataframe: {}'.format(df.shape))

utils.write_metadata(os.path.join(LOG_DIR, "metadata.tsv"), df['label'])

with tf.Session() as sess:
    # assign the tensor that we want to visualize to the embedding variable
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.metadata_path = "metadata.tsv"
    embedding_var = tf.Variable(np.array(embedding_data), name="T2S_embedding")
    embedding.tensor_name = embedding_var.name
    sess.run(embedding_var.initializer)
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    # config.model_checkpoint_path = os.path.join(LOG_DIR, 't2s.ckpt')
    embedding.sprite.image_path = "sprite.jpg"
    embedding.sprite.single_image_dim.extend([utils.img_w, utils.img_h])
    projector.visualize_embeddings(summary_writer, config)

    saver = tf.train.Saver({"T2S_embedding": embedding_var})
    saver.save(sess, os.path.join(LOG_DIR, 't2s_features'))
