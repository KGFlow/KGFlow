import numpy as np
from KGFlow.data.kg import KG


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
        return np.random.choice(kg.num_entities, len(source_entities), replace=True)

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
