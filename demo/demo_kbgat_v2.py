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

units_list = [30, 50]
entity_embedding_size = 50
relation_embedding_size = 50
num_heads = 8

drop_rate = 0.0
l2_coe = 1e-3


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

        entity_feature_list, relation_features = self.gat1([(h_index, r_index, t_index), entity_features,
                                                            relation_features], training=training)
        entity_features = tf.add_n(entity_feature_list)
        entity_features += self.dense(entity_embeddings)

        return entity_features, relation_features


model = KBGATModel()


@tf.function(experimental_relax_shapes=True)
def forward(indices, training=False):
    return model(inputs=indices, training=training)


@tf.function(experimental_relax_shapes=True)
def compute_distmult_loss(entity_features, relation_features, batch_source, batch_r, batch_target, batch_neg_target):
    # entity_features = tf.nn.l2_normalize(entity_features, axis=-1)
    # relation_features = tf.nn.l2_normalize(relation_features, axis=-1)

    s = tf.gather(entity_features, batch_source, axis=0)
    r = tf.gather(relation_features, batch_r, axis=0)
    t = tf.gather(entity_features, batch_target, axis=0)
    nt = tf.gather(entity_features, batch_neg_target, axis=0)

    pos_logits = tf.reduce_sum(s * r * t, axis=-1)
    neg_logits = tf.reduce_sum(s * r * nt, axis=-1)

    losses = tf.maximum(0.0, neg_logits - pos_logits)
    # losses = tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=tf.concat([pos_logits, neg_logits], axis=0),
    #     labels=tf.concat([tf.ones_like(pos_logits), tf.zeros_like(neg_logits)], axis=0)
    # )
    loss = tf.reduce_mean(losses)
    return loss


# margin = 2.0
# train_batch_size = 200
test_batch_size = 200
learning_rate = 1e-2
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

for epoch in range(1, 10001):

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

        # kernel_vals = [var for var in tape.watched_variables() if "kernel" in var.name]
        # l2_losses = [tf.nn.l2_loss(kernel_var) for kernel_var in kernel_vals]
        # loss += tf.add_n(l2_losses) * l2_coe

        loss += (tf.nn.l2_loss(model.entity_embeddings) + tf.nn.l2_loss(model.relation_embeddings)) * l2_coe

    vars = tape.watched_variables()
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    if epoch % 10 == 0:
        print("epoch = {}\tloss = {}".format(epoch, loss))

    if epoch % 100 == 0:

        entity_embeddings, relation_embeddings = forward([test_kg.h, test_kg.r, test_kg.t], training=False)
        # entity_embeddings = tf.nn.l2_normalize(entity_embeddings, axis=-1)
        # relation_embeddings = tf.nn.l2_normalize(relation_embeddings, axis=-1)

        tiled_entity_embeddings = tf.tile(tf.expand_dims(entity_embeddings, axis=0),
                                          [test_batch_size, 1, 1])

        for target_entity_type in ["head", "tail"]:
            all_ranks = []
            for test_step, (batch_h, batch_r, batch_t) in enumerate(
                    tf.data.Dataset.from_tensor_slices((test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):

                if target_entity_type == "tail":
                    batch_source = batch_h
                    batch_target = batch_t
                else:
                    batch_source = batch_t
                    batch_target = batch_h

                s = tf.gather(entity_embeddings, batch_source, axis=0)
                r = tf.gather(relation_embeddings, batch_r, axis=0)

                tiled_sr = tf.tile(tf.expand_dims(s * r, axis=1), [1, entity_embeddings.shape[0], 1])

                if tf.shape(tiled_entity_embeddings)[0] != tf.shape(batch_h)[0]:
                    tiled_entity_embeddings = tf.tile(tf.expand_dims(entity_embeddings, axis=0),
                                                      [tf.shape(batch_h)[0], 1, 1])

                distmult = tf.reduce_sum(tiled_sr * tiled_entity_embeddings, axis=-1)

                ranks = tf.argsort(tf.argsort(-distmult, axis=1), axis=1).numpy()
                target_ranks = ranks[np.arange(len(batch_target)), batch_target.numpy()]

                all_ranks.extend(target_ranks)

            all_ranks = np.array(all_ranks) + 1

            mean_rank = np.mean(all_ranks)
            mrr = np.mean(1 / all_ranks)
            hits_1 = np.sum(all_ranks == 1)
            hits_3 = np.sum(all_ranks <= 3)
            hits_10 = np.sum(all_ranks <= 10)
            hits_100 = np.sum(all_ranks <= 100)
            hits_1000 = np.sum(all_ranks <= 1000)

            print(
                "epoch = {}\ttarget_entity_type = {}\tmean_rank = {}\tmrr = {}\t"
                "hits@1 = {}\thits@3 = {}\thits@10 = {}\thits@100 = {}\thits@1000 = {}".format(
                    epoch, target_entity_type, mean_rank, mrr,
                    hits_1, hits_3, hits_10, hits_100, hits_1000))
