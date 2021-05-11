import numpy as np
from KGFlow.data.kg import KG
import tensorflow as tf
import collections


def entity_negative_sampling(source_entities, relations, kg, target_entity_type="tail", filtered=False):
    """

    :param source_entities:
    :param relations:
    :param kg: KG
    :param target_entity_type: "head" | "tail"
    :param filtered:
    :return:
    """

    if not filtered:
        return np.random.randint(0, kg.num_entities, len(source_entities))

    if target_entity_type == "tail":
        source_relation_target_dict = kg.hrt_dict
    else:
        source_relation_target_dict = kg.trh_dict

    neg_entities = []
    for source, relation in zip(source_entities, relations):
        pos_entities = source_relation_target_dict[int(source)][int(relation)]
        while True:
            neg_entity = np.random.randint(0, kg.num_entities)
            if neg_entity not in pos_entities:
                neg_entities.append(neg_entity)
                break

    return np.array(neg_entities)


class EntityNegativeSampler:
    def __init__(self, kg: KG):
        self.kg = kg
        self.entity_set = set(list(range(kg.num_entities)))

    def random_sampling(self, batch_size, num_neg=1):

        return np.random.randint(0, self.kg.num_entities, num_neg * batch_size)

    def target_sampling(self, source_entities, relations, target_entity_type, num_neg=1, filtered=True):
        if not filtered:
            return self.random_sampling(len(source_entities), num_neg)

        if target_entity_type == "tail":
            sro_dict = self.kg.hrt_dict
        else:
            sro_dict = self.kg.trh_dict

        neg_target_list = []
        for s, r in zip(source_entities, relations):

            pos_o_set = sro_dict[int(s)][int(r)]
            neg_target = []
            while len(neg_target) < num_neg:
                entity = np.random.randint(0, self.kg.num_entities)
                if int(entity) not in pos_o_set:
                    neg_target.append(entity)
            neg_target_list.append(neg_target)

        neg_target_entities = np.stack(neg_target_list, axis=-1).reshape([-1])
        return neg_target_entities

    def target_indices_sampling(self, source_entities, relations, target_entity_type, num_neg=1, filtered=True):
        neg_o = self.target_sampling(source_entities, relations, target_entity_type, num_neg, filtered)
        tiled_s = np.tile(source_entities, [num_neg])
        tiled_r = np.tile(relations, [num_neg])
        if target_entity_type == "tail":
            indices = [tiled_s, tiled_r, neg_o]
        else:
            indices = [neg_o, tiled_r, tiled_s]
        indices = np.stack(indices)
        return indices

    def indices_sampling(self, h, r, t, num_neg=1, filtered=False):
        neg_indices_h = self.target_indices_sampling(t, r, target_entity_type="head", num_neg=num_neg, filtered=filtered)
        neg_indices_t = self.target_indices_sampling(h, r, target_entity_type="tail", num_neg=num_neg, filtered=filtered)
        neg_entities = np.concatenate([neg_indices_h, neg_indices_t], axis=-1)
        return neg_entities


class NeighborSampler:
    def __init__(self, kg: KG):
        self.head_unique = kg.head_unique
        self.num_head = len(self.head_unique)

        self.indices_dict = {}
        print("load entities and neighbors")
        for h in self.head_unique:
            triples = []
            rt_dict = kg.hrt_dict[h]
            for r, v in rt_dict.items():
                for t in v:
                    triples.append([h, r, t])
            triples = np.array(triples)
            graph_indices = triples.T
            self.indices_dict[h] = graph_indices

    def sample(self, batch_size, depth: int = 1, k: int = None, ratio: int = None):

        sampled_h = np.random.choice(self.head_unique, batch_size, replace=False)
        return self.sample_from_h(sampled_h, depth, k, ratio)

    def sample_from_h(self, sampled_h, depth: int = 1, k: int = None, ratio: int = None):
        if k is not None and ratio is not None:
            raise Exception("you should provide either k or ratio, not both of them")

        indices = [sampled_h]
        all_indices = []

        visit = set()
        for i in range(depth):
            next_h = [t for t in set(indices[-1]) if t not in visit and t in self.head_unique]
            if not next_h:
                break
            indices = []
            for h in next_h:
                indices.append(self.indices_dict[h])
            indices = np.concatenate(indices, axis=-1)
            all_indices.append(indices)
            if i < depth - 1:
                visit.update(next_h)

        all_indices = np.concatenate(all_indices, axis=-1)
        return all_indices
