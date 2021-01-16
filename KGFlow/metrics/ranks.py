import tensorflow as tf
import numpy as np


def compute_ranks_by_scores(scores, target_indices, direction='ASCENDING'):
    """
    compute ranks based on the scores and target indices
    :param scores: shape: [batch_size, num_sort]
    :param target_indices: shape: [num_sort], dtype: int
    :param direction: The direction in which to sort the scores (`'ASCENDING'` or
      `'DESCENDING'`).
    :return: target ranks, shape: [batch_size]
    """
    all_ranks = tf.argsort(tf.argsort(scores, axis=-1, direction=direction), axis=-1)
    target_ranks = tf.gather_nd(all_ranks, tf.stack([tf.range(tf.shape(target_indices)[0]),
                                                     tf.cast(target_indices, tf.int32)], axis=-1))
    return target_ranks + 1


def compute_hits(ranks, nums):
    if isinstance(nums, int):
        return np.mean(np.array(ranks) <= nums)
    return [np.mean(np.array(ranks) <= num) for num in nums]


def compute_mean_rank(ranks):
    return np.mean(ranks)


def compute_mean_reciprocal_rank(ranks):
    return np.mean(1 / ranks)
