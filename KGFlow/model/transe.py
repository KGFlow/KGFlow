# coding=utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np


class TransE(keras.Model):
    def __init__(self, n_entity, n_relation, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = tf.Variable(
            tf.random.uniform((n_entity, embedding_size), -6 / np.sqrt(embedding_size), 6 / np.sqrt(embedding_size))
        )
        self.relation_embeddings = tf.Variable(
            tf.random.uniform((n_relation, embedding_size), -6 / np.sqrt(embedding_size), 6 / np.sqrt(embedding_size))
        )

    def embed_norm(self, embeddings, indices):

        # if embedding table is smaller, normalizing first is more efficient
        if embeddings.shape[0] < len(indices):
            return tf.nn.embedding_lookup(tf.nn.l2_normalize(embeddings, axis=-1), indices)
        else:
            return tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, indices), axis=-1)

    def embed_norm_entities(self, entities):
        return self.embed_norm(self.entity_embeddings, entities)

    def embed_norm_relations(self, relations):
        return self.embed_norm(self.relation_embeddings, relations)

    def call(self, inputs, target_entity_type="tail", training=None, mask=None):
        """

        :param inputs: [source_entities, relations]
        :param target_entity_type: "head" | "tail"
        :param training:
        :param mask:
        :return:
        """

        source, r = inputs

        embedded_s = self.embed_norm_entities(source)
        embedded_r = self.embed_norm_relations(r)

        if target_entity_type == "tail":
            translated = embedded_s + embedded_r
        else:
            translated = embedded_s - embedded_r

        return translated

