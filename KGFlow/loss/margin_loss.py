import tensorflow as tf


def margin_loss(pos_scores, neg_scores, margin=0.0):
    losses = tf.maximum(margin + pos_scores - neg_scores, 0.0)
    loss = tf.reduce_mean(losses)
    return loss