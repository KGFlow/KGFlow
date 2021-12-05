import tensorflow as tf


class DistMult(tf.keras.Model):
    def __init__(self, norm_ord=2, epsilon=1e-9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_ord = norm_ord
        self.epsilon = epsilon

    def call(self, inputs, training=None, mask=None):
        h, r, t = inputs[0], inputs[1], inputs[2]

        scores = distmult(h, r, t, self.norm_ord, self.epsilon)

        return scores


def distmult(h, r, t, norm_ord=2, epsilon=1e-9):
    if norm_ord:
        h = h / tf.maximum(tf.norm(h, ord=norm_ord, axis=-1, keepdims=True), epsilon)
        r = r / tf.maximum(tf.norm(r, ord=norm_ord, axis=-1, keepdims=True), epsilon)
        t = t / tf.maximum(tf.norm(t, ord=norm_ord, axis=-1, keepdims=True), epsilon)

    scores = -tf.reduce_sum(h * r * t, axis=-1)

    return scores
