# coding=utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.model.transe import TransE
from KGFlow.dataset.wn18 import WN18Dataset
from KGFlow.utils.sampling_utils import entity_negative_sampling
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

train_kg, test_kg, valid_kg, entity_indexer, relation_indexer = WN18Dataset().load_data()

margin = 2
embedding_size = 20
train_batch_size = 200
test_batch_size = 100

optimizer = keras.optimizers.Adam(learning_rate=1e-2)


def compute_distance(a, b):
    # return tf.reduce_sum(tf.math.square(a - b), axis=-1)
    return tf.reduce_sum(tf.math.abs(a - b), axis=-1)


model = TransE(train_kg.num_entities, train_kg.num_relations, embedding_size)

for epoch in range(10000):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(10000).batch(train_batch_size)):

        target_entity_type = ["head", "tail"][np.random.randint(0, 2)]
        if target_entity_type == "tail":
            batch_source = batch_h
            batch_target = batch_t
        else:
            batch_source = batch_t
            batch_target = batch_h
        batch_neg_target = entity_negative_sampling(batch_source, batch_r, train_kg, target_entity_type, filtered=True)

        with tf.GradientTape() as tape:
            embedded_neg_target = model.embed_norm_entities(batch_neg_target)
            embedded_target = model.embed_norm_entities(batch_target)
            translated = model([batch_source, batch_r], target_entity_type)

            pos_dis = compute_distance(translated, embedded_target)
            neg_dis = compute_distance(translated, embedded_neg_target)

            losses = tf.maximum(margin + pos_dis - neg_dis, 0.0)
            loss = tf.reduce_mean(losses)

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 100 == 0:
            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))

    if epoch % 10 == 0:

        normed_entity_embeddings = tf.math.l2_normalize(model.entity_embeddings, axis=-1)

        for target_entity_type in ["head", "tail"]:
            mean_ranks = []
            for test_step, (batch_h, batch_r, batch_t) in enumerate(
                    tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):

                if target_entity_type == "tail":
                    batch_source = batch_h
                    batch_target = batch_t
                else:
                    batch_source = batch_t
                    batch_target = batch_h

                translated = model([batch_source, batch_r], target_entity_type=target_entity_type)

                tiled_entity_embeddings = tf.tile(tf.expand_dims(normed_entity_embeddings, axis=0),
                                                  [batch_h.shape[0], 1, 1])
                tiled_translated = tf.tile(tf.expand_dims(translated, axis=1),
                                           [1, normed_entity_embeddings.shape[0], 1])
                dis = compute_distance(tiled_translated, tiled_entity_embeddings)

                ranks = tf.argsort(tf.argsort(dis, axis=1), axis=1).numpy()
                target_ranks = ranks[np.arange(len(batch_target)), batch_target.numpy()]
                mean_ranks.extend(target_ranks)

            print("epoch = {}\ttarget_entity_type = {}\tmean_rank = {}".format(epoch, target_entity_type,
                                                                               np.mean(mean_ranks)))
