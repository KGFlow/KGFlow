# coding=utf-8
from KGFlow.dataset.common import CommonDataset


class WN18Dataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="WN18",
                         download_urls=[
                             "https://github.com/CrawlScript/tf_kge_data/raw/main/WN18.zip"
                         ],
                         download_file_name="WN18.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)