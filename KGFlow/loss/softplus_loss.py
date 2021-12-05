import tensorflow as tf


def _softplus_loss(scores, labels):
    """
    :param scores:
    :param labels: pos sample: +1, neg_sample: -1
    :return: loss, shape: []
    """
    labels = tf.reshape(tf.cast(labels, dtype=tf.float32), tf.shape(scores))
    losses = tf.nn.softplus(scores * labels)
    return tf.reduce_mean(losses)


def softplus_loss(pos_scores, neg_scores):
    pos_loss = _softplus_loss(pos_scores, tf.ones_like(pos_scores))
    neg_loss = _softplus_loss(neg_scores, -tf.ones_like(neg_scores))
    loss = (pos_loss + neg_loss) / 2
    return loss