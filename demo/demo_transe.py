# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.model.transe import TransE, compute_distance, transe_ranks
from KGFlow.dataset.wn18 import WN18Dataset

from KGFlow.dataset.fb15k import FB15kDataset, FB15k237Dataset
from KGFlow.utils.sampling_utils import entity_negative_sampling
from KGFlow.metrics.ranks import compute_hits, compute_mean_rank, compute_mean_reciprocal_rank
from KGFlow.utils.rank_utils import get_filter_dict

# train_kg, test_kg, valid_kg, entity2id, relation2id = WN18Dataset().load_data()
train_kg, test_kg, valid_kg, entity2id, relation2id, entity_init_embeddings, relation_init_embeddings = FB15k237Dataset().load_data()

embedding_size = 20
train_n_batch = 100
train_batch_size = train_kg.num_triples // train_n_batch
test_batch_size = 200
margin = 2.0
distance_norm = 1

learning_rate = 1e-2
l2_coe = 0.0

filter = True
filter_dict = get_filter_dict(test_kg, [train_kg, valid_kg]) if filter else None

E = kgf.RandomInitEmbeddings(train_kg.num_entities, train_kg.num_relations, embedding_size,
                             initializer=tf.keras.initializers.glorot_uniform())

model = TransE(E.entity_embeddings, E.relation_embeddings)


@tf.function
def forward(inputs, target_entity_type, training=False):
    return model(inputs, target_entity_type=target_entity_type, training=training)


compute_loss = tf.function(model.compute_loss)
compute_ranks = transe_ranks

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(1, 10001):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(300000).batch(train_batch_size)):

        target_entity_type = ["head", "tail"][np.random.randint(0, 2)]
        if target_entity_type == "tail":
            batch_source = batch_h
            batch_target = batch_t
        else:
            batch_source = batch_t
            batch_target = batch_h

        with tf.GradientTape() as tape:
            batch_neg_target = entity_negative_sampling(batch_source, batch_r, train_kg, target_entity_type,
                                                        filtered=False)

            embedded_neg_target = model.embed_norm_entities(batch_neg_target)
            embedded_target = model.embed_norm_entities(batch_target)

            translated = forward([batch_source, batch_r], target_entity_type, training=True)

            pos_dis = compute_distance(translated, embedded_target, distance_norm)
            neg_dis = compute_distance(translated, embedded_neg_target, distance_norm)

            loss = compute_loss(pos_dis, neg_dis, margin=margin, l2_coe=l2_coe)

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 200 == 0:
            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))

    if epoch % 200 == 0:

        normed_entity_embeddings = tf.math.l2_normalize(model.entity_embeddings, axis=-1)

        for target_entity_type in ["head", "tail"]:
            ranks = []
            for test_step, (batch_h, batch_r, batch_t) in enumerate(
                    tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):
                target_ranks = compute_ranks(batch_h, batch_r, batch_t, forward, normed_entity_embeddings,
                                             target_entity_type, distance_norm=distance_norm,
                                             filter_list=filter_dict[target_entity_type])
                ranks.append(target_ranks)

            ranks = tf.concat(ranks, axis=0)

            mean_rank = compute_mean_rank(ranks)
            mrr = compute_mean_reciprocal_rank(ranks)
            hits_1, hits_3, hits_10, hits_100, hits_1000 = compute_hits(ranks, [1, 3, 10, 100, 1000])

            print(
                "epoch = {}\ttarget_entity_type = {}\tMR = {}\tMRR = {}\t"
                "Hits@1 = {}\tHits@3 = {}\tHits@10 = {}\tHits@100 = {}\tHits@1000 = {}".format(
                    epoch, target_entity_type, mean_rank, mrr,
                    hits_1, hits_3, hits_10, hits_100, hits_1000))
