import tensorflow as tf
from KGFlow.metrics.convkb import ConvKBLayer
from KGFlow.metrics.ranks import compute_ranks_by_scores


class ConvKB(tf.keras.Model):
    def __init__(self, entity_embeddings, relation_embeddings, num_filters=64, kernel_size=1, activation=tf.nn.relu,
                 drop_rate=0.0, use_bn=False):
        super().__init__()
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings

        self.convkb = ConvKBLayer(num_filters, kernel_size, activation, drop_rate, use_bn)

    def call(self, inputs, training=None, mask=None):
        h_index, r_index, t_index = inputs[0], inputs[1], inputs[2]

        h = tf.nn.embedding_lookup(self.entity_embeddings, h_index)
        r = tf.nn.embedding_lookup(self.relation_embeddings, r_index)
        t = tf.nn.embedding_lookup(self.entity_embeddings, t_index)

        scores = self.convkb([h, r, t], training=training)

        return scores

    def compute_loss(self, scores, labels, activation=tf.nn.softplus):
        loss = convkb_loss(scores, labels, activation)
        return loss


def convkb_loss(scores, labels, activation=tf.nn.softplus):
    """
    loss for ConvKB
    :param scores:
    :param labels: pos sample: +1, neg_sample: -1
    :param activation:
    :return: loss, shape: []
    """
    scores = tf.reshape(scores, [-1])
    labels = tf.reshape(tf.cast(labels, dtype=tf.float32), [-1])
    losses = activation(scores * labels)
    return tf.reduce_mean(losses)


def convkb_ranks(batch_h, batch_r, batch_t, num_entities, convkb_model, target_entity_type):
    _batch_size = tf.shape(batch_h)[0]
    if target_entity_type == "tail":
        batch_source = batch_h
        batch_target = batch_t
    else:
        batch_source = batch_t
        batch_target = batch_h

    tiled_s = tf.reshape(tf.tile(tf.expand_dims(batch_source, axis=-1), [1, num_entities]), [-1])
    tiled_r = tf.reshape(tf.tile(tf.expand_dims(batch_r, axis=-1), [1, num_entities]), [-1])
    tiled_o = tf.tile(tf.range(num_entities), [_batch_size])

    if target_entity_type == "tail":
        tiled_indices = [tiled_s, tiled_r, tiled_o]
    else:
        tiled_indices = [tiled_o, tiled_r, tiled_s]

    scores = convkb_model(tiled_indices)
    scores = tf.reshape(scores, [-1])
    split_size = tf.tile([tf.shape(scores)[0] // _batch_size], [_batch_size])
    scores = tf.stack(tf.split(scores, split_size))

    return compute_ranks_by_scores(scores, batch_target)
