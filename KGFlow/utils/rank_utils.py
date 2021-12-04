import tensorflow as tf
import numpy as np
from KGFlow.data.kg import KG


# def compute_ranks_by_scores(scores, target_indices):
#
#     all_ranks = tf.argsort(tf.argsort(scores, axis=-1), axis=-1)
#     target_ranks = tf.gather_nd(all_ranks, tf.stack([tf.range(tf.shape(target_indices)[0]),
#                                                      tf.cast(target_indices, tf.int32)], axis=-1))
#     return target_ranks + 1

def compute_ranks_by_scores(scores, target_indices):
    """
    compute ranks based on the scores and target indices
    :param scores: shape: [batch_size, num_sort]
    :param target_indices: shape: [num_sort], dtype: int
    :return: target ranks, shape: [batch_size]
    """

    targets = tf.gather_nd(scores, tf.stack([tf.range(tf.shape(target_indices)[0]),
                                             tf.cast(target_indices, tf.int32)], axis=-1))
    targets = tf.expand_dims(targets, axis=-1)
    target_ranks = tf.reduce_sum(tf.cast(scores < targets, tf.float32), axis=-1)

    return target_ranks + 1


def compute_ranks_by_model(batch_h, batch_r, batch_t, num_entities, model, target_entity_type, batch_filter_list):
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

    scores = model(tiled_indices)
    scores = tf.reshape(scores, [-1])

    split_size = tf.tile([tf.shape(scores)[0] // _batch_size], [_batch_size])
    scores = tf.stack(tf.split(scores, split_size))

    if batch_filter_list:
        filter_indices = []
        for i, filters in enumerate(batch_filter_list):
            for entity in filters:
                filter_indices.append([i, entity])
        if filter_indices:
            scores = tf.tensor_scatter_nd_update(scores, filter_indices, [float("inf")] * len(filter_indices))

    return compute_ranks_by_scores(scores, batch_target)


def compute_ranks(test_kg: KG, model, target_entity_type, test_batch_size=10, filter_dict=None):
    """
    compute ranks
    :param test_kg: KG
    :param model: a score function, input: indices = [h_index, r_index, t_index]
    :param target_entity_type: 'head' | 'tail'
    :param filter_dict: a dict for filter ranks by get_filter_dict function
    :return: target ranks, shape: [test_kg.num_triples]
    """
    filter_list = filter_dict[target_entity_type] if filter_dict else None

    ranks = []
    for i, (batch_h, batch_r, batch_t) in enumerate(tf.data.Dataset.from_tensor_slices(
            (test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):
        batch_filter_list = filter_list[i * test_batch_size: (i + 1) * test_batch_size] if filter_list else None
        target_ranks = compute_ranks_by_model(batch_h, batch_r, batch_t, test_kg.num_entities, model,
                                              target_entity_type, batch_filter_list)
        ranks.append(target_ranks)

    ranks = tf.concat(ranks, axis=0)
    return ranks


# def compute_ranks_by_model_cache(batch_h, batch_r, batch_t, num_entities, model, target_entity_type,
#                                  batch_filter_indices):
#     _batch_size = tf.shape(batch_h)[0]
#     if target_entity_type == "tail":
#         batch_source = batch_h
#         batch_target = batch_t
#     else:
#         batch_source = batch_t
#         batch_target = batch_h
#
#     tiled_s = tf.reshape(tf.tile(tf.expand_dims(batch_source, axis=-1), [1, num_entities]), [-1])
#     tiled_r = tf.reshape(tf.tile(tf.expand_dims(batch_r, axis=-1), [1, num_entities]), [-1])
#     tiled_o = tf.tile(tf.range(num_entities), [_batch_size])
#
#     if target_entity_type == "tail":
#         tiled_indices = [tiled_s, tiled_r, tiled_o]
#     else:
#         tiled_indices = [tiled_o, tiled_r, tiled_s]
#
#     scores = model(tiled_indices)
#     scores = tf.reshape(scores, [-1])
#
#     split_size = tf.tile([tf.shape(scores)[0] // _batch_size], [_batch_size])
#     scores = tf.stack(tf.split(scores, split_size))
#
#     if batch_filter_indices:
#         updates = tf.numpy_function(lambda x: np.zeros([x]) + np.inf, [len(batch_filter_indices)], tf.float32)
#         scores = tf.tensor_scatter_nd_update(scores, batch_filter_indices, tf.cast(updates, dtype=tf.float32))
#
#     return compute_ranks_by_scores(scores, batch_target)
#
#
# def compute_ranks_cache(test_kg: KG, model, target_entity_type, test_batch_size=10, filter_cache=None):
#     """
#     compute ranks
#     :param test_kg: KG
#     :param model: a score function, input: indices = [h_index, r_index, t_index]
#     :param target_entity_type: 'head' | 'tail'
#     :param filter_cache: a dict for filter ranks by get_filter_dict function
#     :return: target ranks, shape: [test_kg.num_triples]
#     """
#
#     ranks = []
#     for i, (batch_h, batch_r, batch_t) in enumerate(tf.data.Dataset.from_tensor_slices(
#             (test_kg.h, test_kg.r, test_kg.t)).batch(test_batch_size)):
#         batch_filter_indices = filter_cache[
#             CACHE_KEY_FILTER_INDICES.format(target_entity_type, i)] if filter_cache else None
#         target_ranks = compute_ranks_by_model_cache(batch_h, batch_r, batch_t, test_kg.num_entities, model,
#                                               target_entity_type, batch_filter_indices)
#         ranks.append(target_ranks)
#
#     ranks = tf.concat(ranks, axis=0)
#     return ranks


def get_filter_dict(test_kg: KG, filter_kgs) -> dict:
    """
    filter_kgs: KG or KG list
    """

    kg_list = [filter_kgs] if isinstance(filter_kgs, KG) else list(filter_kgs)

    head_filter_list = []
    tail_filter_list = []
    for h, r, t in test_kg.graph_triples:
        head_filters = set()
        tail_filters = set()
        for kg in kg_list + [test_kg]:
            head_filters |= kg.trh_dict.get(int(t), {}).get(int(r), set())
            tail_filters |= kg.hrt_dict.get(int(h), {}).get(int(r), set())

        head_filters = head_filters - {int(h)}
        tail_filters = tail_filters - {int(t)}
        head_filters = list(head_filters)
        tail_filters = list(tail_filters)

        head_filter_list.append(head_filters)
        tail_filter_list.append(tail_filters)

    filter_dict = {"head": head_filter_list, "tail": tail_filter_list}
    return filter_dict


# CACHE_KEY_FILTER_INDICES = "filter_type_{}_batch_{}"


# def build_filter_indices_cache(filter_dict, batch_size):
#     cache = {}
#     for filter_type in ["head", "tail"]:
#         filter_list = filter_dict[filter_type]
#         for batch in range((len(filter_list) - 1) // batch_size + 1):
#             filter_indices = []
#             batch_filter_list = filter_list[batch * batch_size:(batch + 1) * batch_size]
#             for idx, filters in enumerate(batch_filter_list):
#                 for entity in filters:
#                     filter_indices.append([idx, entity])
#             cache[CACHE_KEY_FILTER_INDICES.format(filter_type, batch)] = filter_indices
#
#     return cache
