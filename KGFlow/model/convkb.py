import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Dense
from KGFlow.loss.losses import softplus_loss
from KGFlow.evaluation import compute_ranks


class ConvKB(tf.keras.Model):
    def __init__(self, num_filters, kernel_size=1, activation=tf.nn.relu, drop_rate=0.0, use_bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1d = Conv1D(num_filters, kernel_size)
        self.activation = activation if activation else lambda x: x
        self.drop_out = Dropout(drop_rate)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()
        self.kernel = Dense(1, use_bias=False)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [batch_embedded_h, batch_embedded_r, batch_embedded_t]
        :param training:
        :param mask:
        :return: a score Tensor which shape is (batch_size, 1)
        """
        # h, r, t = inputs[0], inputs[1], inputs[2]

        f = tf.stack(inputs, axis=-1)  # (batch * dim * 3)
        if self.use_bn:
            f = self.bn1(f, training=training)
        f = self.conv1d(f)
        if self.use_bn:
            f = self.bn2(f, training=training)
        f = self.activation(f)
        f = tf.reshape(f, [tf.shape(f)[0], -1])
        f = self.drop_out(f, training=training)

        scores = self.kernel(f)

        return -scores


class ConvKBEmb(tf.keras.Model):
    def __init__(self, entity_embeddings, relation_embeddings, num_filters=64, kernel_size=1, activation=tf.nn.relu,
                 drop_rate=0.0, use_bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        self.convkb = ConvKB(num_filters, kernel_size, activation, drop_rate, use_bn)

    def call(self, inputs, training=None, mask=None):
        h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]

        h = tf.nn.embedding_lookup(self.entity_embeddings, h_index)
        r = tf.nn.embedding_lookup(self.relation_embeddings, r_index)
        t = tf.nn.embedding_lookup(self.entity_embeddings, t_index)

        scores = self.convkb([h, r, t], training=training)

        return scores

    def compute_loss(self, pos_scores, neg_scores):
        loss = convkb_loss(pos_scores, neg_scores)
        return loss


convkb_loss = softplus_loss
convkb_ranks = compute_ranks

# def convkb_ranks(batch_h, batch_r, batch_t, num_entities, convkb_model, target_entity_type):
#     _batch_size = tf.shape(batch_h)[0]
#     if target_entity_type == "tail":
#         batch_source = batch_h
#         batch_target = batch_t
#     else:
#         batch_source = batch_t
#         batch_target = batch_h
#
#     tiled_s = tf.reshape(tf.tile(tf.expand_dims(batch_source, axis=-1), [1, num_entities]), [-1])
#     tiled_r = tf.reshape(tf.tile(tf.expand_dims(batch_r, axis=-1), [1, num_entities]), [-1])
#     tiled_o = tf.tile(tf.range(num_entities), [_batch_size])
#
#     if target_entity_type == "tail":
#         tiled_indices = [tiled_s, tiled_r, tiled_o]
#     else:
#         tiled_indices = [tiled_o, tiled_r, tiled_s]
#
#     scores = convkb_model(tiled_indices)
#     scores = tf.reshape(scores, [-1])
#     split_size = tf.tile([tf.shape(scores)[0] // _batch_size], [_batch_size])
#     scores = tf.stack(tf.split(scores, split_size))
#
#     return compute_ranks_by_scores(scores, batch_target)
