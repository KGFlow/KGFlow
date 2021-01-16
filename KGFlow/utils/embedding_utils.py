import tensorflow as tf
import numpy as np


class RandomInitEmbeddings:
    def __init__(self, num_entities, num_relations, embedding_size, initializer=None, relation_initializer=None):
        if initializer:
            self.initializer = initializer
        else:
            self.initializer = tf.keras.initializers.truncated_normal(stddev=np.sqrt(1 / embedding_size))
        self.relation_initializer = relation_initializer if relation_initializer else self.initializer

        self.entity_embeddings = tf.Variable(self.initializer([num_entities, embedding_size]),
                                             name="entity_embeddings")
        self.relation_embeddings = tf.Variable(self.relation_initializer([num_relations, embedding_size]),
                                               name="relation_embeddings")

    def __call__(self, *args, **kwargs):
        return self.entity_embeddings, self.relation_embeddings
