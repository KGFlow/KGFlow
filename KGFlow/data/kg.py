# coding=utf-8

from tqdm import tqdm
import pickle as pkl
import numpy as np


class KG(object):
    def __init__(self, h, r, t, entity2id: dict = None, relation2id: dict = None):
        self.h = h
        self.r = r
        self.t = t
        self.graph_indices = np.stack([h, r, t], axis=0)
        self.graph_triples = self.graph_indices.T

        self.entity2id = entity2id
        self.relation2id = relation2id
        self._id2relation = None
        self._id2entity = None

        self._entities = None
        self._relations = None

        self.hrt_dict = self.build_triple_dict(self.h, self.r, self.t)
        self._htr_dict = None
        self._trh_dict = None

        self._head_unique = None
        self._tail_unique = None

    def build_triple_dict(self, source_entities, relations, target):
        source_relation_target_dict = {}

        for source, relation, target in tqdm(zip(source_entities, relations, target)):
            relation_target_dict = source_relation_target_dict.setdefault(int(source), dict())
            relation_target_dict.setdefault(int(relation), set()).add(int(target))

        return source_relation_target_dict

    def sample(self):
        pass

    def save_as_pkl(self, save_path):
        with open(save_path, "wb") as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)

    @property
    def id2relation(self):
        if not self._id2relation:
            self._id2relation = {v: k for k, v in self.relation2id.items()}
        return self._id2relation

    @property
    def id2entity(self):
        if not self._id2entity:
            self._id2entity = {v: k for k, v in self.entity2id.items()}
        return self._id2entity

    @property
    def htr_dict(self):
        if not self._htr_dict:
            self._htr_dict = self.build_triple_dict(self.h, self.t, self.r)
        return self._htr_dict

    @property
    def trh_dict(self):
        if not self._trh_dict:
            self._trh_dict = self.build_triple_dict(self.t, self.r, self.h)
        return self._trh_dict

    @property
    def head_unique(self):
        if not self._head_unique:
            self._head_unique = sorted(list(set(list(self.h))))
        return self._head_unique

    @property
    def tail_unique(self):
        if not self._tail_unique:
            self._tail_unique = sorted(list(set(list(self.t))))
        return self._tail_unique

    @property
    def entities(self):
        if not self._entities:
            self._entities = sorted(self.id2entity.keys())
        return self._entities

    @property
    def relations(self):
        if not self._relations:
            self._relations = sorted(self.id2relation.keys())
        return self._relations

    @property
    def num_entities(self):
        return len(self.entity2id)

    @property
    def num_relations(self):
        return len(self.relation2id)

    @property
    def num_triples(self):
        return len(self.h)

    def __str__(self):
        return "KG: entities => {}\trelations => {} triples => {}".format(self.num_entities, self.num_relations,
                                                                          self.num_triples)
