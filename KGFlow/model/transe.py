# coding=utf-8

import tensorflow as tf
import numpy as np
from KGFlow.evaluation import compute_ranks_by_scores
from KGFlow.loss.losses import margin_loss


class TransE(tf.keras.Model):
    def __init__(self, entity_embeddings, relation_embeddings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

    @staticmethod
    def embed_norm(embeddings, indices):

        # if embedding table is smaller, normalizing first is more efficient
        if embeddings.shape[0] < len(indices):
            return tf.nn.embedding_lookup(tf.nn.l2_normalize(embeddings, axis=-1), indices)
        else:
            return tf.nn.l2_normalize(tf.nn.embedding_lookup(embeddings, indices), axis=-1)

    def embed_norm_entities(self, entities):
        return self.embed_norm(self.entity_embeddings, entities)

    def embed_norm_relations(self, relations):
        relation_embeddings = tf.concat([self.relation_embeddings, -self.relation_embeddings], axis=0)
        return self.embed_norm(relation_embeddings, relations)

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

    def compute_loss(self, pos_scores, neg_scores, margin=0.0, l2_coe=0.0):
        loss = transe_loss(pos_scores, neg_scores, margin)
        if l2_coe > 0.0:
            loss += tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables if
                              "kernel" in var.name or "embedding" in var.name] + [0.0]) * l2_coe
        return loss


transe_loss = margin_loss


def compute_distance(a, b, norm=1):
    if norm == 1:
        return tf.reduce_sum(tf.math.abs(a - b), axis=-1)
    if norm == 2:
        return tf.reduce_sum(tf.math.square(a - b), axis=-1)


def transe_batch_ranks(batch_h, batch_r, batch_t, transe_model, entity_embeddings, target_entity_type, distance_norm=1,
                       filter_list=None):
    if target_entity_type == "tail":
        batch_source = batch_h
        batch_target = batch_t
    else:
        batch_source = batch_t
        batch_target = batch_h

    translated = transe_model([batch_source, batch_r], target_entity_type=target_entity_type, training=False)

    # tiled_entity_embeddings = tf.tile(tf.expand_dims(entity_embeddings, axis=0),
    #                                   [batch_h.shape[0], 1, 1])
    # tiled_translated = tf.tile(tf.expand_dims(translated, axis=1),
    #                            [1, entity_embeddings.shape[0], 1])
    tiled_entity_embeddings = tf.expand_dims(entity_embeddings, axis=0)
    tiled_translated = tf.expand_dims(translated, axis=1)
    scores = compute_distance(tiled_translated, tiled_entity_embeddings, distance_norm)

    if filter_list:
        filter_indices = []
        for i, filters in enumerate(filter_list):
            for entity in filters:
                filter_indices.append([i, entity])
        if filter_indices:
            scores = tf.tensor_scatter_nd_update(scores, filter_indices, [float("inf")] * len(filter_indices))

    return compute_ranks_by_scores(scores, batch_target)


def transe_ranks(test_kg, transe_model, entity_embeddings, target_entity_type, batch_size=200, distance_norm=1,
                 filter_list=None):
    ranks = []
    for test_step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(batch_size)):
        target_ranks = transe_batch_ranks(batch_h, batch_r, batch_t, transe_model, entity_embeddings,
                                          target_entity_type, distance_norm,
                                          filter_list[test_step * batch_size: (test_step + 1) * batch_size]
                                          if filter_list else None)
        ranks.append(target_ranks)

    ranks = tf.concat(ranks, axis=0)

    return ranks
