# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
import KGFlow as kgf
from KGFlow.model.capse import CapsE, capse_loss, capse_ranks
from KGFlow.dataset.fb15k import FB15k237Dataset
from KGFlow.utils import EntityNegativeSampler, get_filter_dict
from KGFlow.evaluation import evaluate_rank_scores

# data_dict = WN18Dataset().load_data()
data_dict = FB15k237Dataset().load_data()
train_kg, test_kg, valid_kg, entity2id, relation2id = (data_dict[name] for name in
                                                       ["train_kg", "test_kg", "valid_kg", "entity2id", "relation2id"])
entity_init_embeddings, relation_init_embeddings = (data_dict.get(name, None) for name in
                                                    ["entity_embeddings", "relation_embeddings"])

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
    entity_embeddings = tf.Variable(entity_init_embeddings, name="entity_embeddings")
    relation_embeddings = tf.Variable(relation_init_embeddings, name="relation_embeddings")
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


compute_loss = tf.function(capse_loss)
compute_ranks = capse_ranks


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

            print("epoch = {}\ttarget_entity_type = {}".format(epoch, target_entity_type))
            res_scores = evaluate_rank_scores(ranks, ["mr", "mrr", "hits"], [1, 3, 10, 100, 1000])
            print(res_scores)
