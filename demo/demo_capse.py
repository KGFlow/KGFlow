# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import KGFlow as kgf
from KGFlow.model.capse import CapsE
from KGFlow.dataset.fb15k import FB15k237Dataset
from KGFlow.utils.sampling_utils import EntityNegativeSampler
from KGFlow.utils.rank_utils import get_filter_dict, compute_ranks
from KGFlow.metrics.ranks import compute_hits, compute_mean_rank, compute_mean_reciprocal_rank


train_kg, test_kg, valid_kg, entity2id, relation2id, entity_embeddings, relation_embeddings = FB15k237Dataset().load_data()

init_embedding = True
filter = True
train_filtered = False
num_filters = 50
num_neg = 10
train_batch_size = 256
test_batch_size = 1

learning_rate = 1e-4
drop_rate = 0.5
l2_coe = 0.02

filter_dict = get_filter_dict(test_kg, [train_kg, valid_kg]) if filter else None
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

if init_embedding:
    embedding_size = 100
    entity_embeddings = tf.Variable(entity_embeddings, name="entity_embeddings")
    relation_embeddings = tf.Variable(relation_embeddings, name="relation_embeddings")
else:
    embedding_size = 20
    E = kgf.RandomInitEmbeddings(train_kg.num_entities, train_kg.num_relations, embedding_size)
    entity_embeddings, relation_embeddings = E()


model = CapsE(entity_embeddings,
              relation_embeddings,
              sequence_length=3,
              initialization=[],
              embedding_size=embedding_size,
              filter_size=1,
              num_filters=num_filters,
              vocab_size=-1,
              iter_routing=1,
              batch_size=train_batch_size,
              num_outputs_secondCaps=1,
              vec_len_secondCaps=10,
              useConstantInit=False
              )

sampler = EntityNegativeSampler(train_kg)


# @tf.function
def forward(batch_indices, training=False):
    return model(batch_indices, training=training)


# @tf.function
def compute_loss(pos_scores, neg_scores):

    pos_loss = model.compute_loss(pos_scores, tf.ones_like(pos_scores))
    neg_loss = model.compute_loss(neg_scores, -tf.ones_like(neg_scores))

    loss = pos_loss + neg_loss
    return loss


for epoch in range(0, 10001):
    for step, (batch_h, batch_r, batch_t) in enumerate(
            tf.data.Dataset.from_tensor_slices((train_kg.h, train_kg.r, train_kg.t)).
                    shuffle(300000).batch(train_batch_size)):

        with tf.GradientTape() as tape:
            batch_neg_indices = sampler.indices_sampling(batch_h, batch_r, batch_t, num_neg=num_neg,
                                                         filtered=train_filtered)

            pos_scores = forward([batch_h, batch_r, batch_t], training=True)
            neg_scores = forward(batch_neg_indices, training=True)

            loss = compute_loss(pos_scores, neg_scores)

        vars = tape.watched_variables()
        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if step % 20 == 0:
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
