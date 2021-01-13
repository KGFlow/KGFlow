# coding=utf-8
from KGFlow.dataset.common import CommonDataset


class FB15kDataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="FB15k",
                         download_urls=[
                             "https://github.com/CrawlScript/tf_kge_data/raw/main/FB15k.zip"
                         ],
                         download_file_name="FB15k.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)