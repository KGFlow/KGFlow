# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.model.transe import TransE
from KGFlow.dataset.wn18 import WN18Dataset
from KGFlow.utils.sampling_utils import entity_negative_sampling

train_kg, test_kg, valid_kg, entity2id, relation2id = WN18Dataset().load_data()

num_entities = len(entity2id)
num_relations = len(relation2id)

units_list = [150, 300]
entity_embedding_size = 50
relation_embedding_size = 50
num_heads = 8

drop_rate = 0.0
l2_coe = 5e-4


class KBGATModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = tf.Variable(tf.random.truncated_normal([num_entities, entity_embedding_size],
                                                                        stddev=np.sqrt(1 / entity_embedding_size)))
        self.relation_embeddings = tf.Variable(tf.random.truncated_normal([num_relations, relation_embedding_size],
                                                                          stddev=np.sqrt(1 / relation_embedding_size)))

        self.gat0 = kgf.model.KBGAT(units=units_list[0], num_heads=num_heads)
        self.gat1 = kgf.model.KBGAT(units=units_list[1], num_heads=num_heads)

        self.dense = keras.layers.Dense(units_list[-1])
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, inputs, training=None, mask=None):

        h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]

        entity_embeddings = self.dropout(self.entity_embeddings, training=training)
        entity_feature_list, relation_features = self.gat0([(h_index, r_index, t_index), entity_embeddings,
                                                       self.relation_embeddings], training=training)

        entity_features = tf.concat(entity_feature_list, axis=-1)
        entity_features = self.dropout(entity_features, training=training)

        entity_feature_list, relation_features = self.gat1([(h_index, r_index, t_index), entity_features, relation_features],
                                                       training=training)
        entity_features = tf.add_n(entity_feature_list)
        entity_features += self.dense(entity_embeddings)

        return entity_features, relation_features


model = KBGATModel()


@tf.function(experimental_relax_shapes=True)
def forward(batch_indices, training=False):
    return model(inputs=batch_indices, training=training)


class EntityAndNeighborSampler:
    def __init__(self, kg):
        self.head_unique = kg.head_unique
        self.num_head = len(self.head_unique)

        self.indices_dict = {}
        print("load entities and neighbors")
        for h in self.head_unique:
            triples = []
            rt_dict = kg.hrt_dict[h]
            for r, v in rt_dict.items():
                for t in v:
                    triples.append([h, r, t])
            triples = np.array(triples)
            graph_indices = triples.T
            self.indices_dict[h] = graph_indices

    def sample(self, batch_size):
        sampled_h = np.random.choice(self.head_unique, batch_size, replace=False)
        indices = []
        for h in sampled_h:
            indices.append(self.indices_dict[h])
        indices = np.concatenate(indices, axis=-1)

        return indices


# def sample_entities_and_neighbors(kg, batch_size):
#     sampled_h = np.random.choice(kg.head_unique, batch_size, replace=False)
#     triples = []
#     for h in range(sampled_h):
#         rt_dict = kg.hrt_dict[h]
#         for r, v in rt_dict.items():
#             for t in v:
#                 triples.append([h, r, t])
#     triples = np.array(triples)
#     graph_indices = triples.T
#
#     return graph_indices


@tf.function(experimental_relax_shapes=True)
def compute_loss(entity_features, relation_features, source, relation, target, neg_target):
    s = tf.gather(entity_features, source, axis=0)
    t = tf.gather(entity_features, target, axis=0)
    nt = tf.gather(entity_features, neg_target, axis=0)
    r = tf.gather(relation_features, relation, axis=0)
    pos_logits = tf.reduce_sum(s*r*t, axis=-1)
    neg_logits = tf.reduce_sum(s*r*nt, axis=-1)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.concat([pos_logits, neg_logits], axis=0),
        labels=tf.concat([tf.ones_like(pos_logits), tf.zeros_like(neg_logits)], axis=0)
    )
    loss = tf.reduce_mean(losses)
    return loss


entities_and_neighbors_sampler = EntityAndNeighborSampler(train_kg)


margin = 2.0
train_batch_size = 200
learning_rate = 1e-2
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

for step in range(10000):
    batch_h, batch_r, batch_t = entities_and_neighbors_sampler.sample(train_batch_size)
    target_entity_type = ["head", "tail"][np.random.randint(0, 2)]
    if target_entity_type == "tail":
        batch_source = batch_h
        batch_target = batch_t
    else:
        batch_source = batch_t
        batch_target = batch_h
    batch_neg_target = entity_negative_sampling(batch_source, batch_r, train_kg, target_entity_type, filtered=True)

    with tf.GradientTape() as tape:

        entity_features, relation_features = forward([batch_h, batch_r, batch_t], training=True)

        loss = compute_loss(entity_features, relation_features, batch_source, batch_r, batch_target, batch_neg_target)

        kernel_vals = [var for var in tape.watched_variables() if "kernel" in var.name]
        l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
        loss += tf.add_n(l2_losses) * l2_coe

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if step % 20 == 0:
        print("step = {}\tloss = {}".format(step, loss))
