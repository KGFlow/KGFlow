# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import KGFlow as kgf
from KGFlow.model.transe import TransE
from KGFlow.dataset.wn18 import WN18Dataset
from KGFlow.utils.sampling_utils import entity_negative_sampling
import time

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

        entity_feature_list, relation_features = self.gat1(
            [(h_index, r_index, t_index), entity_features, relation_features],
            training=training)
        entity_features = tf.add_n(entity_feature_list)
        entity_features += self.dense(entity_embeddings)

        return entity_features, relation_features


model = KBGATModel()


@tf.function(experimental_relax_shapes=True)
def forward(batch_indices, training=False):
    return model(inputs=batch_indices, training=training)


#
# class NeighborSampler:
#     def __init__(self, kg):
#         self.head_unique = kg.head_unique
#         self.num_head = len(self.head_unique)
#
#         self.indices_dict = {}
#         print("load entities and neighbors")
#         for h in self.head_unique:
#             triples = []
#             rt_dict = kg.hrt_dict[h]
#             for r, v in rt_dict.items():
#                 for t in v:
#                     triples.append([h, r, t])
#             triples = np.array(triples)
#             graph_indices = triples.T
#             self.indices_dict[h] = graph_indices
#
#     def sample(self, batch_size, depth: int = 1, k: int = None, ratio: int = None):
#         if k is not None and ratio is not None:
#             raise Exception("you should provide either k or ratio, not both of them")
#
#         sampled_h = np.random.choice(self.head_unique, batch_size, replace=False)
#         indices = []
#         for h in sampled_h:
#             indices.append(self.indices_dict[h])
#         indices = np.concatenate(indices, axis=-1)
#         all_indices = [indices]
#
#         if depth > 1:
#             visit = set(sampled_h)
#             for i in range(1, depth):
#                 next_h = [t for t in set(indices[-1]) if t not in visit and t in self.head_unique]
#                 if not next_h:
#                     break
#                 indices = []
#                 for h in next_h:
#                     indices.append(self.indices_dict[h])
#                 indices = np.concatenate(indices, axis=-1)
#                 all_indices.append(indices)
#                 if i < depth - 1:
#                     visit.update(next_h)
#
#         all_indices = np.concatenate(all_indices, axis=-1)
#
#         return all_indices


class NeighborSampler:
    def __init__(self, kg, depth=1):
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

    def sample(self, batch_size, depth: int = 1, k: int = None, ratio: int = None):

        sampled_h = np.random.choice(self.head_unique, batch_size, replace=False)
        return self.sample_from_h(sampled_h, depth, k, ratio)

    def sample_from_h(self, sampled_h, depth: int = 1, k: int = None, ratio: int = None):
        if k is not None and ratio is not None:
            raise Exception("you should provide either k or ratio, not both of them")

        indices = [sampled_h]
        all_indices = []

        visit = set()
        for i in range(depth):
            next_h = [t for t in set(indices[-1]) if t not in visit and t in self.head_unique]
            if not next_h:
                break
            indices = []
            for h in next_h:
                indices.append(self.indices_dict[h])
            indices = np.concatenate(indices, axis=-1)
            all_indices.append(indices)
            if i < depth - 1:
                visit.update(next_h)

        all_indices = np.concatenate(all_indices, axis=-1)
        return all_indices


@tf.function(experimental_relax_shapes=True)
def compute_loss(entity_features, relation_features, batch_source, batch_r, batch_target, batch_neg_target):
    s = tf.gather(entity_features, batch_source, axis=0)
    r = tf.gather(relation_features, batch_r, axis=0)
    t = tf.gather(entity_features, batch_target, axis=0)
    nt = tf.gather(entity_features, batch_neg_target, axis=0)

    pos_logits = tf.reduce_sum(s * r * t, axis=-1)
    neg_logits = tf.reduce_sum(s * r * nt, axis=-1)
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=tf.concat([pos_logits, neg_logits], axis=0),
        labels=tf.concat([tf.ones_like(pos_logits), tf.zeros_like(neg_logits)], axis=0)
    )
    loss = tf.reduce_mean(losses)
    return loss


neighbors_sampler = NeighborSampler(train_kg)

margin = 2.0
train_batch_size = 200
test_batch_size = 200
learning_rate = 1e-2
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(1, 10001):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(10000).batch(train_batch_size)):

        start = time.time()

        target_entity_type = ["head", "tail"][np.random.randint(0, 2)]
        if target_entity_type == "tail":
            batch_source = batch_h
            batch_target = batch_t
        else:
            batch_source = batch_t
            batch_target = batch_h
        batch_neg_target = entity_negative_sampling(batch_source, batch_r, train_kg, target_entity_type, filtered=True)

        sampled_h = np.concatenate([batch_source, batch_target, batch_neg_target], axis=-1)
        batch_train_indices = neighbors_sampler.sample_from_h(sampled_h, depth=2)

        print("sample_token_time: {}".format(time.time() - start))

        with tf.GradientTape() as tape:

            entity_features, relation_features = forward(batch_train_indices, training=True)

            loss = compute_loss(entity_features, relation_features, batch_source, batch_r, batch_target,
                                batch_neg_target)

            kernel_vals = [var for var in tape.watched_variables() if "kernel" in var.name]
            l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
            loss += tf.add_n(l2_losses) * l2_coe

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 1 == 0:
            print("step = {}\tloss = {}".format(step, loss))

    if epoch % 10 == 0:

        entity_embeddings, relation_embeddings = forward([test_kg.h, test_kg.r, test_kg.t], training=False)
        h = tf.gather(entity_embeddings, test_kg.h, axis=0)
        r = tf.gather(relation_embeddings, test_kg.r, axis=0)
        t = tf.gather(entity_embeddings, test_kg.t, axis=0)

        for target_entity_type in ["head", "tail"]:
            mean_ranks = []

            if target_entity_type == "tail":
                source = h
                target = t
            else:
                source = t
                target = h

            tiled_entity_embeddings = tf.tile(tf.expand_dims(entity_embeddings, axis=0),
                                              [tf.shape(h)[0], 1, 1])
            tiled_sr = tf.tile(tf.expand_dims(source * r, axis=1),
                                       [1, entity_embeddings.shape[0], 1])
            distmult = tiled_sr * tiled_entity_embeddings

            ranks = tf.argsort(tf.argsort(-distmult, axis=1), axis=1).numpy()
            target_ranks = ranks[np.arange(len(target)), target.numpy()]

            mean_ranks.extend(target_ranks)

            print("epoch = {}\ttarget_entity_type = {}\tmean_rank = {}".format(epoch, target_entity_type,
                                                                               np.mean(mean_ranks)))

        # for target_entity_type in ["head", "tail"]:
        #     mean_ranks = []
        #     for test_step, (batch_h, batch_r, batch_t) in enumerate(
        #             tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):
        #
        #         if target_entity_type == "tail":
        #             batch_source = batch_h
        #             batch_target = batch_t
        #         else:
        #             batch_source = batch_t
        #             batch_target = batch_h
        #         sampled_h = set(np.concatenate([batch_source, batch_target], axis=-1))
        #         batch_test_indices = neighbors_sampler.sample_from_h(sampled_h, depth=2)
        #
        #         entity_features, relation_features = forward(batch_test_indices, training=False)
        #
        #         tiled_entity_embeddings = tf.tile(tf.expand_dims(normed_entity_embeddings, axis=0),
        #                                           [batch_h.shape[0], 1, 1])
        #         tiled_translated = tf.tile(tf.expand_dims(translated, axis=1),
        #                                    [1, normed_entity_embeddings.shape[0], 1])
        #         dis = compute_distance(tiled_translated, tiled_entity_embeddings)
        #
        #         ranks = tf.argsort(tf.argsort(dis, axis=1), axis=1).numpy()
        #         target_ranks = ranks[np.arange(len(batch_target)), batch_target.numpy()]
        #         mean_ranks.extend(target_ranks)
        #
        #     print("epoch = {}\ttarget_entity_type = {}\tmean_rank = {}".format(epoch, target_entity_type,
        #                                                                        np.mean(mean_ranks)))
