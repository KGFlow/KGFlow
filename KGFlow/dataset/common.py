# coding=utf-8
from typing import Tuple

import os
from tqdm import tqdm
import numpy as np

from tf_geometric.data.dataset import DownloadableDataset
import json

from KGFlow.data.kg import KG


class CommonDataset(DownloadableDataset):

    def _read_name2id(self, name2id_path) -> dict:
        print("reading name2id_info: ", name2id_path)
        name2id = {}
        with open(name2id_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                name = items[0]
                id = int(items[1])
                name2id[name] = id
        return name2id

    def _read_triples(self, triple_path, entity2id: dict, relation2id: dict) -> KG:
        triples = []
        print("reading triples: ", triple_path)
        with open(triple_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if len(line) == 0:
                    continue
                head_entity, tail_entity, relation = line.split()
                triple = [
                    entity2id[head_entity],
                    relation2id[relation],
                    entity2id[tail_entity]
                ]
                triples.append(triple)
        triples = np.array(triples)
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        return KG(heads, relations, tails, entity2id, relation2id)

    def process(self) -> Tuple[KG, KG, KG, dict, dict]:

        data_dir = os.path.join(self.raw_root_path, self.dataset_name)

        test_triple_path = os.path.join(data_dir, "test.txt")
        train_triple_path = os.path.join(data_dir, "train.txt")
        valid_triple_path = os.path.join(data_dir, "valid.txt")

        entity_path = os.path.join(data_dir, "entity2id.txt")
        relation_path = os.path.join(data_dir, "relation2id.txt")

        entity2id = self._read_name2id(entity_path)
        relation2id = self._read_name2id(relation_path)

        train_kg = self._read_triples(train_triple_path, entity2id, relation2id)
        test_kg = self._read_triples(test_triple_path, entity2id, relation2id)
        valid_kg = self._read_triples(valid_triple_path, entity2id, relation2id)

        return train_kg, test_kg, valid_kg, entity2id, relation2id
