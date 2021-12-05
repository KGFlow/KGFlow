from tqdm import tqdm
import tensorflow as tf
import numpy as np
from KGFlow.metrics.ranking import *


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

    tiled_s = tf.repeat(batch_source, num_entities)
    tiled_r = tf.repeat(batch_r, num_entities)
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


def evaluate_rank_scores(ranks, metrics, k_list=(1, 10, 100)):
    res_scores = {}
    for metric in metrics:
        if metric.lower() in ("mr", "mean_rank"):
            score_func = mean_rank_score
        elif metric.lower() in ("mrr", "mean_reciprocal_rank"):
            score_func = mean_reciprocal_rank_score
        elif metric.lower() == "hits":
            score_func = lambda x: hits_score(x, k_list)
        else:
            raise Exception("Not Found Metric : {}".format(metric))

        res_score = score_func(ranks)
        if metric == "hits":
            for k, score in zip(k_list, res_score):
                res_scores["hits@{}".format(k)] = score
        else:
            res_scores[metric] = res_score

    return res_scores
