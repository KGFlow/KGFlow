from tqdm import tqdm
import tensorflow as tf
import numpy as np

from KGFlow.metrics.ranking import *


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
