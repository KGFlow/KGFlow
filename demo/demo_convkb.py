# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.model.convkb import ConvKB, convkb_ranks
from KGFlow.dataset.fb15k import FB15kDataset, FB15k237Dataset
from KGFlow.utils.sampling_utils import entity_negative_sampling
from KGFlow.metrics.ranks import compute_hits, compute_mean_rank, compute_mean_reciprocal_rank

# train_kg, test_kg, valid_kg, entity2id, relation2id = FB15kDataset().load_data()
train_kg, test_kg, valid_kg, entity2id, relation2id, entity_embeddings, relation_embeddings = FB15k237Dataset().load_data()

init_embedding = True
num_filters = 64
train_batch_size = 8000
test_batch_size = 10

learning_rate = 1e-4
drop_rate = 0.0
l2_coe = 1e-3

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

if init_embedding:
    entity_embeddings = tf.Variable(entity_embeddings, name="entity_embeddings")
    relation_embeddings = tf.Variable(relation_embeddings, name="relation_embeddings")
    model = ConvKB(entity_embeddings, relation_embeddings, num_filters, drop_rate=drop_rate)
else:
    embedding_size = 20
    E = kgf.RandomInitEmbeddings(train_kg.num_entities, train_kg.num_relations, embedding_size)
    model = ConvKB(E.entity_embeddings, E.relation_embeddings, num_filters, drop_rate=drop_rate)


@tf.function
def forward(batch_indices, training=False):
    return model(batch_indices, training=training)


@tf.function
def compute_loss(pos_scores, neg_scores):
    loss = model.compute_loss(tf.concat([pos_scores, neg_scores], axis=0),
                              tf.concat([tf.ones_like(pos_scores), -tf.ones_like(neg_scores)], axis=0),
                              activation=tf.nn.softplus, l2_coe=l2_coe)
    return loss


compute_ranks = tf.function(convkb_ranks)


for epoch in range(1, 10001):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(200000).batch(train_batch_size)):

        with tf.GradientTape() as tape:
            target_entity_type = ["head", "tail"][np.random.randint(0, 2)]
            if target_entity_type == "tail":
                batch_neg_target = entity_negative_sampling(batch_h, batch_r, train_kg, "tail", filtered=False)
                batch_neg_indices = [batch_h, batch_r, batch_neg_target]
            else:
                batch_neg_target = entity_negative_sampling(batch_t, batch_r, train_kg, "head", filtered=False)
                batch_neg_indices = [batch_neg_target, batch_r, batch_t]

            pos_scores = forward([batch_h, batch_r, batch_t], training=True)
            neg_scores = forward(batch_neg_indices, training=True)

            loss = compute_loss(pos_scores, neg_scores)

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

    if epoch % 20 == 0:
        print("epoch = {}\tloss = {}".format(epoch, loss))

    if epoch % 200 == 0:

        for target_entity_type in ["head", "tail"]:
            ranks = []
            for (batch_h, batch_r, batch_t) in tf.data.Dataset.from_tensor_slices(
                    (test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size):
                target_ranks = compute_ranks(batch_h, batch_r, batch_t, test_kg.num_entities, forward, target_entity_type)
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
