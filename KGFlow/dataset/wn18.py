# coding=utf-8
from KGFlow.dataset.common import CommonDataset
from typing import Tuple

import os
import numpy as np
from tqdm import tqdm

from KGFlow.data.kg import KG


class WN18Dataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="WN18",
                         download_urls=[
                             "https://github.com/KGFlow/kge_datasets/raw/main/WN18.zip"
                         ],
                         download_file_name="WN18.zip",
                         cache_name=None,  # "cache.p",
                         dataset_root_path=dataset_root_path)

    def process(self):
        return self.common_process(triple_type="htr")


class WN18RRDataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="WN18RR",
                         download_urls=[
                             "https://github.com/KGFlow/kge_datasets/raw/main/WN18RR.zip"
                         ],
                         download_file_name="WN18RR.zip",
                         cache_name=None,  # "cache.p",
                         dataset_root_path=dataset_root_path)

    def process(self) -> dict:

        # train_kg, test_kg, valid_kg, entity2id, relation2id = self.common_process("hrt")
        data_dict = self.common_process("hrt")

        entity2vec_path = os.path.join(self.data_dir, "entity2vec.txt")
        relation2vec_path = os.path.join(self.data_dir, "relation2vec.txt")

        data_dict["entity_embeddings"] = self._read_name2vec(entity2vec_path)
        data_dict["relation_embeddings"] = self._read_name2vec(relation2vec_path)

        # return train_kg, test_kg, valid_kg, entity2id, relation2id, entity_embeddings, relation_embeddings
        return data_dict
