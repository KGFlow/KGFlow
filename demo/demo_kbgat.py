# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.model.gat import KBGAT
from KGFlow.dataset.fb15k import FB15kDataset, FB15k237Dataset
from KGFlow.utils import entity_negative_sampling
from KGFlow.evaluation.ranking import compute_ranks_by_scores
from KGFlow.evaluation import evaluate_rank_scores

# data_dict = WN18Dataset().load_data()
data_dict = FB15k237Dataset().load_data()
train_kg, test_kg, valid_kg, entity2id, relation2id = (data_dict[name] for name in
                                                       ["train_kg", "test_kg", "valid_kg", "entity2id", "relation2id"])
entity_init_embeddings, relation_init_embeddings = (data_dict.get(name, None) for name in
                                                    ["entity_embeddings", "relation_embeddings"])

num_entities = len(entity2id)
num_relations = len(relation2id)
print(train_kg, test_kg, valid_kg)

init_embedding = True
units_list = [50, 50]
entity_embedding_size = 100
relation_embedding_size = 100
num_heads = 1

drop_rate = 0.0
l2_coe = 1e-3

test_batch_size = 200
learning_rate = 1e-2


if init_embedding:
    entity_embeddings = tf.Variable(entity_init_embeddings, name="entity_embeddings")
    relation_embeddings = tf.Variable(relation_init_embeddings, name="relation_embeddings")
else:
    embedding_size = 20
    E = kgf.RandomInitEmbeddings(train_kg.num_entities, train_kg.num_relations, embedding_size)
    entity_embeddings, relation_embeddings = E()


class KBGATModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        self.gat0 = KBGAT(units=units_list[0], num_heads=num_heads, activation=None, relation_activation=tf.nn.relu)
        self.gat1 = KBGAT(units=units_list[1], num_heads=num_heads, activation=None, relation_activation=None)

        self.dense = keras.layers.Dense(units_list[-1])
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):
        h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]

        E_entity = self.entity_embeddings
        E_relation = self.relation_embeddings

        E_entity = self.dropout(E_entity, training=training)
        entity_feature_list, relation_features = self.gat0([(h_index, r_index, t_index), E_entity,
                                                            E_relation], training=training)

        entity_features = tf.concat(entity_feature_list, axis=-1)
        entity_features = self.dropout(entity_features, training=training)

        entity_feature_list, relation_features = self.gat1([(h_index, r_index, t_index), entity_features,
                                                            relation_features], training=training)
        entity_features = tf.add_n(entity_feature_list)
        entity_features += self.dense(entity_embeddings)

        return entity_features, relation_features


model = KBGATModel()


@tf.function
def forward(indices, training=False):
    return model(inputs=indices, training=training)


@tf.function
def compute_distmult_loss(entity_features, relation_features, batch_source, batch_r, batch_target, batch_neg_target):
    s = tf.gather(entity_features, batch_source, axis=0)
    r = tf.gather(relation_features, batch_r, axis=0)
    t = tf.gather(entity_features, batch_target, axis=0)
    nt = tf.gather(entity_features, batch_neg_target, axis=0)

    pos_logits = tf.reduce_sum(s * r * t, axis=-1)
    neg_logits = tf.reduce_sum(s * r * nt, axis=-1)

    # losses = tf.maximum(0.0, neg_logits - pos_logits)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.concat([pos_logits, neg_logits], axis=0),
        labels=tf.concat([tf.ones_like(pos_logits), tf.zeros_like(neg_logits)], axis=0)
    )
    loss = tf.reduce_mean(losses)
    return loss


optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(1, 100001):

    with tf.GradientTape() as tape:

        entity_features, relation_features = forward(train_kg.graph_indices, training=True)

        target_entity_type = ["head", "tail"][np.random.randint(0, 2)]
        if target_entity_type == "tail":
            source = train_kg.h
            target = train_kg.t
        else:
            source = train_kg.t
            target = train_kg.h
        neg_target = entity_negative_sampling(source, train_kg.r, train_kg, target_entity_type, filtered=False)

        loss = compute_distmult_loss(entity_features, relation_features, source, train_kg.r, target, neg_target)

        kernel_vals = [var for var in tape.watched_variables() if "kernel" in var.name or "embedding" in var.name]
        l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
        loss += tf.add_n(l2_losses) * l2_coe


    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if epoch % 1 == 0:
        print("epoch = {}\tloss = {}".format(epoch, loss))

    if epoch % 20 == 0:

        E_entity, E_relation = forward([test_kg.h, test_kg.r, test_kg.t], training=False)

        tiled_entity_embeddings = tf.tile(tf.expand_dims(E_entity, axis=0),
                                          [test_batch_size, 1, 1])

        for target_entity_type in ["head", "tail"]:
            ranks = []
            for test_step, (batch_h, batch_r, batch_t) in enumerate(
                    tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):

                if target_entity_type == "tail":
                    batch_source = batch_h
                    batch_target = batch_t
                else:
                    batch_source = batch_t
                    batch_target = batch_h

                s = tf.gather(E_entity, batch_source, axis=0)
                r = tf.gather(E_relation, batch_r, axis=0)

                tiled_sr = tf.tile(tf.expand_dims(s * r, axis=1), [1, E_entity.shape[0], 1])

                if tf.shape(tiled_entity_embeddings)[0] != tf.shape(batch_h)[0]:
                    tiled_entity_embeddings = tf.tile(tf.expand_dims(E_entity, axis=0),
                                                      [tf.shape(batch_h)[0], 1, 1])

                distmult = -tf.reduce_sum(tiled_sr * tiled_entity_embeddings, axis=-1)

                target_ranks = compute_ranks_by_scores(distmult, batch_target)
                ranks.append(target_ranks)

            ranks = tf.concat(ranks, axis=0)

            print("epoch = {}\ttarget_entity_type = {}".format(epoch, target_entity_type))
            res_scores = evaluate_rank_scores(ranks, ["mr", "mrr", "hits"], [1, 3, 10, 100, 1000])
            print(res_scores)
