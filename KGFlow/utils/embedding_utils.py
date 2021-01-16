import tensorflow as tf
import numpy as np


class RandomInitEmbeddings:
    def __init__(self, num_entities, num_relations, embedding_size, initializers=None, relation_initializers=None):
        if initializers:
            self.initializers = initializers
        else:
            self.initializers = tf.keras.initializers.truncated_normal(stddev=np.sqrt(1 / embedding_size))
        self.relation_initializers = relation_initializers if relation_initializers else self.initializers

        self.entity_embeddings = tf.Variable(self.initializers([num_entities, embedding_size]),
                                             name="entity_embeddings")
        self.relation_embeddings = tf.Variable(self.relation_initializers([num_relations, embedding_size]),
                                               name="relation_embeddings")

    def __call__(self, *args, **kwargs):
        return self.entity_embeddings, self.relation_embeddings
