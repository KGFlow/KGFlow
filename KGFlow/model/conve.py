import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, SpatialDropout2D


class ConvE(tf.keras.Model):
    def __init__(self, num_filters=16, filter_size=3, activation=tf.nn.relu, drop_rate=0.0, use_bn=False, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = Conv2D(num_filters, filter_size)
        self.height = None
        self.width = None
        self.dense = None

        self.activation = activation if activation else lambda x: x
        self.dropout2d = SpatialDropout2D(drop_rate)
        self.dropout = Dropout(drop_rate)
        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = BatchNormalization()
            self.bn2 = BatchNormalization()

    def build(self, input_shape):
        emb_size = int(input_shape[0][-1])
        self.dense = Dense(emb_size)

        for height in range(int(np.sqrt(emb_size)), 0, -1):
            if emb_size % height == 0:
                self.height = height
                self.width = emb_size // height
                break

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [batch_embedded_h, batch_embedded_r, batch_embedded_t]
        :param training:
        :param mask:
        :return: a score Tensor which shape is (batch_size, 1)
        """
        h, r, t = inputs[0], inputs[1], inputs[2]

        h = tf.reshape(h, [-1, self.height, self.width, 1])
        r = tf.reshape(r, [-1, self.height, self.width, 1])

        x = tf.concat([h, r], axis=1)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout2d(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)
        x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.activation(x)
        scores = tf.reduce_sum(x * t, axis=-1, keepdims=True)

        return scores  # (batch * 1)


class ConvEEmb(tf.keras.Model):
    def __init__(self, entity_embeddings, relation_embeddings, num_filters=16, filter_size=3,
                 activation=tf.nn.relu, drop_rate=0.0, use_bn=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        self.conve = ConvE(num_filters, filter_size, activation, drop_rate, use_bn)

    def call(self, inputs, training=None, mask=None):
        h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]

        h = tf.nn.embedding_lookup(self.entity_embeddings, h_index)
        r = tf.nn.embedding_lookup(self.relation_embeddings, r_index)
        t = tf.nn.embedding_lookup(self.entity_embeddings, t_index)

        scores = self.conve([h, r, t], training=training)

        return scores

    @classmethod
    def compute_loss(cls, scores, labels):
        loss = conve_loss(scores, labels)
        return loss


def conve_loss(scores, labels):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=scores,
        labels=labels
    )
    loss = tf.reduce_mean(losses)
    return loss
