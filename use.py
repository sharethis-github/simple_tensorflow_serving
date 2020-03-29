import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
from tensorflow.saved_model import simple_save

export_dir = "./models/use/00000001"
with tf.Session(graph=tf.Graph()) as sess:
    module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1")
    text_input = tf.placeholder(dtype=tf.string, shape=[None])

    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    embeddings = module(text_input)

    simple_save(sess,
        export_dir,
        inputs={'text': text_input},
        outputs={'embeddings': embeddings},
        legacy_init_op=tf.tables_initializer())
