import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D


class ConvELayer(tf.keras.Model):
    def __init__(self, num_filters, activation=tf.nn.relu, drop_rate=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = Conv2D(16, 3, activation=activation)

        # self.E_e = entity_embeddings
        # self.E_r = relation_embeddings
        # self.E_rr = tf.Variable(tf.keras.initializers.truncated_normal(stddev=np.sqrt(1 / embedding_size))(
        #     [train_kg.num_relations, embedding_size]))
        # self.conv2 = Conv2D(16, 5, activation=activation)

        # self.conv_backend1 = Conv1D(num_filters1, 2, activation=activation)
        # self.dense_backend = Dense(num_filters1, activation=activation)
        # self.conv_backend2 = Conv1D(num_filters2, 2, activation=activation)

        # self.activation = activation if activation else lambda x: x
        # self.drop_out = Dropout(drop_rate)
        # self.use_bn = use_bn
        # if self.use_bn:
        #     self.bn1 = BatchNormalization()
        #     self.bn2 = BatchNormalization()
        self.dense = Dense(num_filters)
        self.kernel = Dense(1)

    def call(self, inputs, training=None, mask=None):
        """

        :param inputs: [batch_embedded_h, batch_embedded_r, batch_embedded_t]
        :param training:
        :param mask:
        :return: a score Tensor which shape is (batch_size, 1)
        """
        _h, _r, _rr, _t = inputs

        h = tf.reshape(_h, [-1, 20, 10, 1])
        r = tf.reshape(_r, [-1, 20, 10, 1])

        x = tf.concat([h, r], axis=-1)
        x = self.conv(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)
        x = tf.nn.relu(x)
        scores = tf.reduce_sum(x * _t, axis=-1, keepdims=True)

        rr = tf.reshape(_rr, [-1, 20, 10, 1])
        t = tf.reshape(_t, [-1, 20, 10, 1])

        x = tf.concat([t, rr], axis=-1)
        x = self.conv(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)
        x = tf.nn.relu(x)
        scores += tf.reduce_sum(x * _h, axis=-1, keepdims=True)

        return scores  # (batch * 1)


def conve_loss(pos_scores, neg_scores):
    pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pos_scores,
        labels=tf.zeros_like(pos_scores)
    )
    neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=neg_scores,
        labels=tf.ones_like(neg_scores)
    )

    loss = tf.reduce_mean(pos_losses) + tf.reduce_mean(neg_losses)
    return loss