# coding=utf-8
from KGFlow.dataset.common import CommonDataset
from typing import Tuple

import os
import numpy as np
from tqdm import tqdm

from KGFlow.data.kg import KG


class FB15kDataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="FB15k",
                         download_urls=[
                             "https://github.com/KGFlow/kge_datasets/raw/main/FB15k.zip"
                         ],
                         download_file_name="FB15k.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)


class FB15k237Dataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="FB15k237",
                         download_urls=[
                             "https://github.com/KGFlow/kge_datasets/raw/main/FB15k237.zip"
                         ],
                         download_file_name="FB15k237.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)

    def process(self) -> Tuple[KG, KG, KG, dict, dict, np.ndarray, np.ndarray]:

        data_dir = os.path.join(self.raw_root_path, self.dataset_name)

        test_triple_path = os.path.join(data_dir, "test.txt")
        train_triple_path = os.path.join(data_dir, "train.txt")
        valid_triple_path = os.path.join(data_dir, "valid.txt")

        entity2id_path = os.path.join(data_dir, "entity2id.txt")
        relation2id_path = os.path.join(data_dir, "relation2id.txt")

        entity2vec_path = os.path.join(data_dir, "entity2vec.txt")
        relation2vec_path = os.path.join(data_dir, "relation2vec.txt")

        entity_embeddings = self._read_name2vec(entity2vec_path)
        relation_embeddings = self._read_name2vec(relation2vec_path)

        entity2id = self._read_name2id(entity2id_path)
        relation2id = self._read_name2id(relation2id_path)

        train_kg = self._read_triples(train_triple_path, entity2id, relation2id, "hrt")
        test_kg = self._read_triples(test_triple_path, entity2id, relation2id, "hrt")
        valid_kg = self._read_triples(valid_triple_path, entity2id, relation2id, "hrt")

        return train_kg, test_kg, valid_kg, entity2id, relation2id, entity_embeddings, relation_embeddings
