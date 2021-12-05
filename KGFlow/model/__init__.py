from KGFlow.model.transe import *
from KGFlow.model.gat import *
from KGFlow.model.gat import *
from KGFlow.model.conve import *
from KGFlow.model.capse import *


# class ModelEmb(tf.keras.Model):
#     def __init__(self, entity_embeddings, relation_embeddings, model, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.entity_embeddings = entity_embeddings
#         self.relation_embeddings = relation_embeddings
#
#         self.model = model
#
#     def call(self, inputs, training=None, mask=None):
#         h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]
#
#         h = tf.nn.embedding_lookup(self.entity_embeddings, h_index)
#         r = tf.nn.embedding_lookup(self.relation_embeddings, r_index)
#         t = tf.nn.embedding_lookup(self.entity_embeddings, t_index)
#
#         scores = self.model([h, r, t], training=training)
#
#         return scores
#
#     @classmethod
#     def compute_loss(cls, scores, labels):
#         pass
