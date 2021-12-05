import tensorflow as tf
import numpy as np


def hits_score(ranks, k_list):
    if isinstance(k_list, int):
        return np.mean(np.array(ranks) <= k_list)
    return [np.mean(np.array(ranks) <= k) for k in k_list]


def mean_rank_score(ranks):
    return np.mean(ranks)


def mean_reciprocal_rank_score(ranks):
    return np.mean(1 / ranks)

