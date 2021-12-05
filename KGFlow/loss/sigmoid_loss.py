import tensorflow as tf


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
