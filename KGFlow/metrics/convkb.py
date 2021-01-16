import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, Dense


class ConvKBLayer(tf.keras.Model):
    def __init__(self, num_filters, kernel_size, activation=tf.nn.relu, drop_rate=0.0, use_bn=True, *args, **kwargs):
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
        h, r, t = inputs

        f = tf.stack([h, r, t], axis=-1)  # (batch * dim * 3)
        if self.use_bn:
            f = self.bn1(f, training=training)
        f = self.conv1d(f)
        if self.use_bn:
            f = self.bn2(f, training=training)
        f = self.activation(f)
        f = tf.reshape(f, [tf.shape(f)[0], -1])
        f = self.drop_out(f, training=training)

        scores = self.kernel(f)

        return scores



