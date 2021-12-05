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
from KGFlow.utils.rank_utils import get_filter_dict
from KGFlow.model import ConvEEmb
from KGFlow.evaluation import evaluate_rank_scores, compute_ranks

# data_dict = WN18Dataset().load_data()
data_dict = FB15k237Dataset().load_data()
train_kg, test_kg, valid_kg, entity2id, relation2id = (data_dict[name] for name in
                                                       ["train_kg", "test_kg", "valid_kg", "entity2id", "relation2id"])
entity_init_embeddings, relation_init_embeddings = (data_dict.get(name, None) for name in
                                                    ["entity_embeddings", "relation_embeddings"])

init_embedding = False
filter = True
num_filters = 16
filter_size = 3
num_neg = 10
train_n_batch = 500
train_batch_size = train_kg.num_triples // train_n_batch
test_batch_size = 5

learning_rate = 1e-4
drop_rate = 0.3
l2_coe = 1e-5

filter_dict = get_filter_dict(test_kg, [train_kg, valid_kg]) if filter else None

if init_embedding:
    entity_embeddings = tf.Variable(entity_init_embeddings, name="entity_embeddings")
    relation_embeddings = tf.Variable(relation_init_embeddings, name="relation_embeddings")
else:
    embedding_size = 200
    E = kgf.RandomInitEmbeddings(train_kg.num_entities, train_kg.num_relations, embedding_size)
    entity_embeddings, relation_embeddings = E()

model = ConvEEmb(entity_embeddings, relation_embeddings, num_filters, filter_size, drop_rate=drop_rate)
sampler = EntityNegativeSampler(train_kg)

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function
def forward(batch_indices, training=False):
    return model(batch_indices, training=training)


@tf.function
def compute_loss(pos_scores, neg_scores):
    # loss = model.compute_loss(tf.concat([pos_scores, neg_scores], axis=0),
    #                           tf.concat([tf.zeros_like(pos_scores), tf.ones_like(neg_scores)], axis=0))
    pos_loss = model.compute_loss(pos_scores, tf.zeros_like(pos_scores))
    neg_loss = model.compute_loss(neg_scores,  tf.ones_like(neg_scores))
    loss = pos_loss + neg_loss
    return loss


for epoch in range(1, 10001):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(300000).batch(train_batch_size)):

        with tf.GradientTape() as tape:
            batch_neg_indices = sampler.indices_sampling(batch_h, batch_r, batch_t, num_neg=num_neg)

            pos_scores = forward([batch_h, batch_r, batch_t], training=True)
            neg_scores = forward(batch_neg_indices, training=True)

            loss = compute_loss(pos_scores, neg_scores)

            l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tape.watched_variables() if "kernel" in var.name])

            l2_loss *= l2_coe
            loss += l2_loss

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        grads, _ = tf.clip_by_global_norm(grads, 0.5)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 100 == 0:
            print("epoch = {}\tstep = {}\tloss = {}".format(epoch, step, loss))

    if epoch % 20 == 0:

        for target_entity_type in ["head", "tail"]:
            ranks = compute_ranks(test_kg, forward, target_entity_type, test_batch_size, filter_dict)

            print("epoch = {}\ttarget_entity_type = {}".format(epoch, target_entity_type))
            res_scores = evaluate_rank_scores(ranks, ["mr", "mrr", "hits"], [1, 3, 10, 100, 1000])
            print(res_scores)
