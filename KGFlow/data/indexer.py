# coding=utf-8


class Indexer(object):
    def __init__(self, id_index_dict, index_id_dict):
        self.id_index_dict = id_index_dict
        self.index_id_dict = index_id_dict

    def __len__(self):
        return len(self.id_index_dict)