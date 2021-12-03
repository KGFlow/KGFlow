# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.dataset.fb15k import FB15kDataset, FB15k237Dataset
from KGFlow.utils.sampling_utils import entity_negative_sampling, EntityNegativeSampler
from KGFlow.utils.rank_utils import get_filter_dict, compute_ranks
from KGFlow.metrics.ranks import compute_hits, compute_mean_rank, compute_mean_reciprocal_rank
from KGFlow.model import ConvELayer

# train_kg, test_kg, valid_kg, entity2id, relation2id = FB15kDataset().load_data()
train_kg, test_kg, valid_kg, entity2id, relation2id, entity_embeddings, relation_embeddings = FB15k237Dataset().load_data()

init_embedding = False
filter = True
train_filtered = False
num_filters = 200
num_neg = 10
train_n_batch = 500
train_batch_size = train_kg.num_triples // train_n_batch
test_batch_size = 10

learning_rate = 1e-4
drop_rate = 0.5
l2_coe = 1e-3

filter_dict = get_filter_dict(test_kg, [train_kg, valid_kg]) if filter else None
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

if init_embedding:
    entity_embeddings = tf.Variable(entity_embeddings, name="entity_embeddings")
    relation_embeddings = tf.Variable(relation_embeddings, name="relation_embeddings")
else:
    embedding_size = 200
    E = kgf.RandomInitEmbeddings(train_kg.num_entities, train_kg.num_relations, embedding_size)
    entity_embeddings, relation_embeddings = E()
    E_rr = tf.Variable(tf.keras.initializers.truncated_normal(stddev=np.sqrt(1 / embedding_size))(
        [train_kg.num_relations, embedding_size]))


model = ConvELayer(num_filters, drop_rate=drop_rate)
sampler = EntityNegativeSampler(train_kg)


# @tf.function
def forward(batch_indices, training=False):
    h_index, r_index, t_index = batch_indices

    h = tf.nn.embedding_lookup(entity_embeddings, h_index)
    r = tf.nn.embedding_lookup(relation_embeddings, r_index)
    rr = tf.nn.embedding_lookup(E_rr, r_index)
    t = tf.nn.embedding_lookup(entity_embeddings, t_index)
    return model([h, r, rr, t], training=training)


# @tf.function
def compute_loss(pos_scores, neg_scores):
    pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pos_scores,
        labels=tf.zeros_like(pos_scores)
    )
    neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=neg_scores,
        labels=tf.ones_like(neg_scores)
    )

    loss = tf.reduce_mean(pos_losses) + tf.reduce_mean(neg_losses)
    return loss


for epoch in range(1, 10001):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(300000).batch(train_batch_size)):

        with tf.GradientTape() as tape:
            batch_neg_indices = sampler.indices_sampling(batch_h, batch_r, batch_t, num_neg=num_neg,
                                                         filtered=train_filtered)

            pos_scores = forward([batch_h, batch_r, batch_t], training=True)
            neg_scores = forward(batch_neg_indices, training=True)

            loss = compute_loss(pos_scores, neg_scores)

            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tape.watched_variables() if "kernel" in var.name])

            l2_loss *= l2_coe
            loss += l2_loss

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        # grads, _ = tf.clip_by_global_norm(grads, 0.5)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 100 == 0:
            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))

    if epoch % 20 == 0:

        for target_entity_type in ["head", "tail"]:
            ranks = compute_ranks(test_kg, forward, target_entity_type, test_batch_size, filter_dict)

            mean_rank = compute_mean_rank(ranks)
            mrr = compute_mean_reciprocal_rank(ranks)
            # hits_1, hits_3, hits_10, hits_100, hits_1000 = compute_hits(ranks, [1, 3, 10, 100, 1000])
            hits_1, hits_10, hits_100 = compute_hits(ranks, [1, 10, 100])
            print(
                "epoch = {}\ttarget_entity_type = {}\tMR = {:f}\tMRR = {:f}\t"
                "Hits@10 = {:f}\tHits@1 = {:f}\tHits@100 = {:f}".format(
                    epoch, target_entity_type, mean_rank, mrr,
                    hits_10, hits_1, hits_100))
