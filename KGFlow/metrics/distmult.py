import tensorflow as tf


@tf.function
def distmult(h, r, t, norm_ord=2, epsilon=1e-9):

    if norm_ord:

        h = h / tf.maximum(tf.norm(h, ord=norm_ord, axis=-1, keepdims=True), epsilon)
        r = r / tf.maximum(tf.norm(r, ord=norm_ord, axis=-1, keepdims=True), epsilon)
        t = t / tf.maximum(tf.norm(t, ord=norm_ord, axis=-1, keepdims=True), epsilon)

    logits = tf.reduce_sum(h * r * t, axis=-1)

    return logits