# coding=utf-8

import tensorflow as tf
import numpy as np
from KGFlow.metrics.ranks import compute_ranks_by_scores


class TransE(tf.keras.Model):
    def __init__(self, entity_embeddings, relation_embeddings, embedding2constant=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        if embedding2constant:
            self.entity_embeddings = tf.constant(self.entity_embeddings)
            self.relation_embeddings = tf.constant(self.relation_embeddings)

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

    def compute_loss(self, pos_logits, neg_logits, margin=0.0, l2_coe=0.0):
        loss = transe_loss(pos_logits, neg_logits, margin)
        if l2_coe > 0.0:
            loss += tf.add_n([tf.nn.l2_loss(var) for var in self.trainable_variables if
                              "kernel" in var.name or "embedding" in var.name]) * l2_coe
        return loss


def transe_loss(pos_logits, neg_logits, margin=0.0):
    losses = tf.maximum(margin + pos_logits - neg_logits, 0.0)
    loss = tf.reduce_mean(losses)
    return loss


def compute_distance(a, b, norm=1):
    if norm == 1:
        return tf.reduce_sum(tf.math.abs(a - b), axis=-1)
    if norm == 2:
        return tf.reduce_sum(tf.math.square(a - b), axis=-1)


def transe_ranks(batch_h, batch_r, batch_t, transe_model, entity_embeddings, target_entity_type, distance_norm=1):

    if target_entity_type == "tail":
        batch_source = batch_h
        batch_target = batch_t
    else:
        batch_source = batch_t
        batch_target = batch_h

    translated = transe_model([batch_source, batch_r], target_entity_type=target_entity_type, training=False)

    tiled_entity_embeddings = tf.tile(tf.expand_dims(entity_embeddings, axis=0),
                                      [batch_h.shape[0], 1, 1])
    tiled_translated = tf.tile(tf.expand_dims(translated, axis=1),
                               [1, entity_embeddings.shape[0], 1])
    dis = compute_distance(tiled_translated, tiled_entity_embeddings, distance_norm)

    return compute_ranks_by_scores(dis, batch_target)
