# coding=utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
from KGFlow.nn import aggregate_neighbors, gcn_mapper, identity_updater, sum_reducer
from KGFlow.nn.kernel.segment import segment_softmax


class GAT(keras.Model):

    def __init__(self, units, activation=tf.nn.relu, att_activation=tf.nn.leaky_relu,
                 drop_rate=0.0, use_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.units = units
        self.activation = activation if activation else lambda x: x

        self.att_activation = att_activation if att_activation else lambda x: x

        self.kernel = tf.keras.layers.Dense(units=self.units, use_bias=False)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.att_kernel = tf.keras.layers.Dense(units=1, activation=self.att_activation)

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = self.add_weight("bias", shape=[self.units], initializer="zeros")

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [(h, r, t), entity_embeddings, relation_embeddings]
        :param training:
        :param mask:
        :return:
        """

        (h_index, r_index, t_index), entity_embeddings, relation_embeddings = inputs
        num_entities = tf.shape(entity_embeddings)[0]

        h = tf.nn.embedding_lookup(entity_embeddings, h_index)
        r = tf.nn.embedding_lookup(relation_embeddings, r_index)
        t = tf.nn.embedding_lookup(entity_embeddings, t_index)
        features = tf.concat([h, r, t], axis=-1)
        features = self.kernel(features)

        att_score = tf.squeeze(self.att_activation(self.att_kernel(features)), axis=-1)
        normed_att_score = segment_softmax(att_score, h_index, num_entities)
        normed_att_score = self.dropout(normed_att_score, training=training)

        neighbor_msg = gcn_mapper(h, features, edge_weight=normed_att_score)
        reduced_msg = sum_reducer(neighbor_msg, h_index, num_nodes=num_entities)
        features = identity_updater(features, reduced_msg)

        # features = aggregate_neighbors(
        #     features, [h_index, t_index], normed_att_score,
        #     gcn_mapper,
        #     sum_reducer,
        #     identity_updater,
        #     num_nodes=num_entities
        # )
        if self.use_bias:
            features += self.bias
        features = self.activation(features)

        return features


class KBGAT(keras.Model):
    """
    Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs
    """

    def __init__(self, units, relation_units=None, num_heads=1, activation=tf.nn.relu,
                 att_activation=tf.nn.leaky_relu,
                 relation_activation=tf.nn.relu, drop_rate=0.0, use_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gats = [GAT(units, activation=activation, att_activation=att_activation, drop_rate=drop_rate,
                         use_bias=use_bias) for _ in range(num_heads)]

        self.relation_kernel = tf.keras.layers.Dense(units=relation_units if relation_units else units,
                                                     activation=relation_activation)
        self.dropout = keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        (h_index, r_index, t_index), entity_embeddings, relation_embeddings = inputs

        entity_feature_list = []
        for i in range(len(self.gats)):
            features = self.gats[i]([(h_index, r_index, t_index), entity_embeddings, relation_embeddings],
                                    training=training)
            entity_feature_list.append(features)

        relation_features = self.relation_kernel(relation_embeddings)

        return entity_feature_list, relation_features
