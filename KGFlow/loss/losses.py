import tensorflow as tf


def margin_loss(pos_scores, neg_scores, margin=0.0):
    losses = tf.maximum(margin + pos_scores - neg_scores, 0.0)
    loss = tf.reduce_mean(losses)
    return loss


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


def _sigmoid_loss(scores, labels):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=scores,
        labels=labels
    )
    loss = tf.reduce_mean(losses)
    return loss


def sigmoid_loss(pos_scores, neg_scores):
    pos_loss = _sigmoid_loss(pos_scores, tf.zeros_like(pos_scores))
    neg_loss = _sigmoid_loss(neg_scores, tf.ones_like(neg_scores))
    loss = (pos_loss + neg_loss) / 2
    return loss
