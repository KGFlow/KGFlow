# coding=utf-8
from KGFlow.dataset.common import CommonDataset
from typing import Tuple

import os
import numpy as np
from tqdm import tqdm

from KGFlow.data.kg import KG


class KinshipDataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="Kinship",
                         download_urls=[
                             ""
                         ],
                         download_file_name="Kinship.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)

    def process(self):
        return self.common_process(triple_type="hrt")

