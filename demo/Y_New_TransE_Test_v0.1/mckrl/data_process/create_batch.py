import torch
import numpy as np
from collections import defaultdict
import os
import time
import queue
import random
import numpy as np
import scipy.sparse as sp
import torch
import pickle
from tqdm import tqdm
import math


class Corpus:
    def __init__(self, args, train_data, validation_data, test_data, entity2id,
                 relation2id, headTailSelector, batch_size, valid_to_invalid_samples_ratio, unique_entities_train,
                 get_2hop=False):
        self.train_triples = train_data[0]

        # multi-topolopy
        adj_indices_topolopy = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
        adj_values_topolopy = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix_topolopy = (adj_indices_topolopy, adj_values_topolopy)

        # multi-context
        # Converting to sparse tensor
        adj_indices_context = torch.LongTensor(
            [train_data[1][1], train_data[1][0]])  # rows and columns
        adj_values_context = torch.LongTensor(train_data[1][2])
        self.train_adj_matrix_context = (adj_indices_context, adj_values_context)

        # adjacency matrix is needed for train_data only, as GAT is trained for
        # training data_pro
        self.validation_triples = validation_data[0]
        self.test_triples = test_data[0]

        self.headTailSelector = headTailSelector  # for selecting random entities
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in self.entity2id.items()}
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in self.relation2id.items()}
        self.batch_size = batch_size
        # ratio of valid to invalid samples per batch for training ConvKB Model
        self.invalid_valid_ratio = int(valid_to_invalid_samples_ratio)
        # self.graph = self.get_graph()
        self.graph_topology = self.get_graph_topology()
        # self.node_neighbors_1hop = self.get_further_neighbors(nbd_size=1)
        # if (get_2hop):
        #     self.node_neighbors_2hop = self.get_further_neighbors(nbd_size=2)
        # if (args.get_2hop):
        #     self.graph = self.get_graph_topology()
        #     self.node_neighbors_2hop = self.get_further_neighbors_topology(nbd_size=2)
        self.unique_entities_train = [self.entity2id[i]
                                      for i in unique_entities_train]
        self.train_indices = np.array(
            list(self.train_triples)).astype(np.int32)
        # These are valid triples, hence all have value 1
        self.train_values = np.array(
            [[1]] * len(self.train_triples)).astype(np.float32)

        self.validation_indices = np.array(
            list(self.validation_triples)).astype(np.int32)
        self.validation_values = np.array(
            [[1]] * len(self.validation_triples)).astype(np.float32)

        self.test_indices = np.array(list(self.test_triples)).astype(np.int32)
        self.test_values = np.array(
            [[1]] * len(self.test_triples)).astype(np.float32)

        self.valid_triples_dict = {j: i for i, j in enumerate(
            self.train_triples + self.validation_triples + self.test_triples)}
        self.valid_triples_dict_raw = {j: i for i, j in enumerate(
            self.train_triples)}
        print("Total triples count {}, training triples {}, validation_triples {}, test_triples {}".format(
            len(self.valid_triples_dict), len(self.train_indices),
            len(self.validation_indices), len(self.test_indices)))

        # For training purpose
        self.batch_indices = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
        self.batch_values = np.empty(
            (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

        # self.adj_train = self.load_graph(args)
        # self.new_1hop, self.new_entity2id, self.multi_edge = self.gen_subgraph(args)
        # self.big_graph_edges = self.load_graph(args)

    def get_iteration_batch(self, iter_num):
        if (iter_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (self.batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            self.batch_size * (iter_num + 1))

            self.batch_indices[:self.batch_size,
            :] = self.train_indices[indices, :]
            self.batch_values[:self.batch_size,
            :] = self.train_values[indices, :]

            last_index = self.batch_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                                        (self.invalid_valid_ratio // 2) + \
                                        (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0],
                               self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

        else:
            last_iter_size = len(self.train_indices) - \
                             self.batch_size * iter_num
            self.batch_indices = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 3)).astype(np.int32)
            self.batch_values = np.empty(
                (last_iter_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * iter_num,
                            len(self.train_indices))
            self.batch_indices[:last_iter_size, :] = self.train_indices[indices, :]
            self.batch_values[:last_iter_size, :] = self.train_values[indices, :]

            last_index = last_iter_size

            if self.invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(self.entity2id), last_index * self.invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
                self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                    self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = i * (self.invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                               self.batch_indices[last_index + current_index, 2]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                    for j in range(self.invalid_valid_ratio // 2):
                        current_index = last_index * \
                                        (self.invalid_valid_ratio // 2) + \
                                        (i * (self.invalid_valid_ratio // 2) + j)

                        while (self.batch_indices[last_index + current_index, 0],
                               self.batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in self.valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(self.entity2id))
                        self.batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        self.batch_values[last_index + current_index, :] = [-1]

                return self.batch_indices, self.batch_values

            return self.batch_indices, self.batch_values

    def get_iteration_batch_nhop(self, current_batch_indices, node_neighbors, batch_size):

        self.batch_indices = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 4)).astype(np.int32)
        self.batch_values = np.empty(
            (batch_size * (self.invalid_valid_ratio + 1), 1)).astype(np.float32)
        indices = random.sample(range(len(current_batch_indices)), batch_size)

        self.batch_indices[:batch_size, :] = current_batch_indices[indices, :]
        self.batch_values[:batch_size, :] = np.ones((batch_size, 1))

        last_index = batch_size

        if self.invalid_valid_ratio > 0:
            random_entities = np.random.randint(
                0, len(self.entity2id), last_index * self.invalid_valid_ratio)

            # Precopying the same valid indices from 0 to batch_size to rest
            # of the indices
            self.batch_indices[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_indices[:last_index, :], (self.invalid_valid_ratio, 1))
            self.batch_values[last_index:(last_index * (self.invalid_valid_ratio + 1)), :] = np.tile(
                self.batch_values[:last_index, :], (self.invalid_valid_ratio, 1))

            for i in range(last_index):
                for j in range(self.invalid_valid_ratio // 2):
                    current_index = i * (self.invalid_valid_ratio // 2) + j

                    self.batch_indices[last_index + current_index,
                                       0] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

                for j in range(self.invalid_valid_ratio // 2):
                    current_index = last_index * \
                                    (self.invalid_valid_ratio // 2) + \
                                    (i * (self.invalid_valid_ratio // 2) + j)

                    self.batch_indices[last_index + current_index,
                                       3] = random_entities[current_index]
                    self.batch_values[last_index + current_index, :] = [0]

            return self.batch_indices, self.batch_values

        return self.batch_indices, self.batch_values

    def get_graph_topology(self):
        graph = {}
        all_tiples = torch.cat([self.train_adj_matrix_topolopy[0].transpose(
            0, 1), self.train_adj_matrix_topolopy[1].unsqueeze(1)], dim=1)

        # graph["out"] = {}
        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()
            if (source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")
        return graph

    def get_graph_context(self):
        graph = {}
        all_tiples = torch.cat([self.train_adj_matrix_context[0].transpose(
            0, 1), self.train_adj_matrix_context[1].unsqueeze(1)], dim=1)
        for data in all_tiples:
            source = data[1].data.item()
            target = data[0].data.item()
            value = data[2].data.item()
            if (source not in graph.keys()):
                graph[source] = {}
                graph[source][target] = value
            else:
                graph[source][target] = value
        print("Graph created")
        return graph

    def bfs(self, graph, source, nbd_size=2):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[source] = 1
        distance[source] = 0
        parent[source] = (-1, -1)

        q = queue.Queue()
        q.put((source, -1))

        while (not q.empty()):
            top = q.get()
            if top[0] in graph.keys():
                for target in graph[top[0]].keys():
                    if (target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target]))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > nbd_size:
                            continue
                        parent[target] = (top[0], graph[top[0]][target])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1

        neighbors = {}
        for target in visit.keys():
            if (distance[target] != nbd_size):
                continue
            # edges = [-1, parent[target][1]]
            relations = []
            entities = [target]
            temp = target
            while (parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if (distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_further_neighbors(self, nbd_size):
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph.keys()))
        for source in self.graph.keys():
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph, source, nbd_size)
            for distance in temp_neighbors.keys():
                if (source in neighbors.keys()):
                    if (distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]

        print("time taken ", time.time() - start_time)
        print("length of {}_hop neighbors dict is {}".format(nbd_size, len(neighbors)))
        return neighbors

    def get_further_neighbors_topology(self, nbd_size=2):
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(self.graph_topology.keys()))
        for source in self.graph_topology.keys():
            # st_time = time.time()
            temp_neighbors = self.bfs(self.graph_topology, source, nbd_size)
            for distance in temp_neighbors.keys():
                if (source in neighbors.keys()):
                    if (distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]

        print("time taken ", time.time() - start_time)
        print("length of {}_hop neighbors dict is {}".format(nbd_size, len(neighbors)))
        return neighbors

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def load_graph(self, args):
        featuregraph_path = args.data + 'big_graph_edges.txt'
        feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
        fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
        fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])),
                             shape=(len(self.id2entity), len(self.id2entity)),
                             dtype=np.float32)
        fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
        nfadj = self.normalize(fadj + sp.eye(fadj.shape[0]))
        nfadj = self.sparse_mx_to_torch_sparse_tensor(nfadj)

        return nfadj

    def gen_adj(self, sample_struct_edges, new_entity2id):
        sedges = np.array(list(sample_struct_edges), dtype=np.int32).reshape(sample_struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                             shape=(len(new_entity2id), len(new_entity2id)),
                             dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        nsadj = self.normalize(sadj + sp.eye(sadj.shape[0]))
        nsadj = self.sparse_mx_to_torch_sparse_tensor(nsadj)
        return nsadj

    def gen_multi_adj(self, rel_dic, new_entity2id):
        multi_edge = []
        for i, rel in enumerate(rel_dic.values()):
            # print(i)
            temp = []
            temp.append((self.gen_adj(rel, new_entity2id[i])))
            multi_edge.append(temp)
        return multi_edge

    def specify_node_update(self, rel_dic):
        rel_entity_set = []
        for tris in rel_dic.values():
            temp = []
            for tri in tris:
                if tri[0] not in temp:
                    temp.append(tri[0])
                if tri[1] not in temp:
                    temp.append(tri[1])
            rel_entity_set.append(temp)
        new_entity2id = {}
        for i, item in enumerate(rel_entity_set):
            new_entity2id[i] = {}
            for j, _ in enumerate(item):
                new_entity2id[i][_] = j
        # update new subgraph entity id
        new_rel_dic = {}
        for key, tris in rel_dic.items():
            temp = []
            for tri in tris:
                _ = [new_entity2id[key][tri[0]], new_entity2id[key][tri[1]]]
                temp.append(_)
            new_rel_dic[key] = np.array(temp)
        return new_entity2id, new_rel_dic

    def get_batch_nhop_neighbors_all_topology(self, args, batch_sources, node_neighbors, nbd_size=2):
        batch_source_triples = []
        print("length of unique_entities ", len(batch_sources))
        count = 0
        for source in batch_sources:
            # randomly select from the list of neighbors
            if source in node_neighbors.keys():
                nhop_list = node_neighbors[source][nbd_size]
                for i, tup in enumerate(nhop_list):
                    if (args.partial_2hop and i >= 2):
                        break
                    count += 1
                    batch_source_triples.append([source, nhop_list[i][0][-1], nhop_list[i][0][0],
                                                 nhop_list[i][1][0]])

        return np.array(batch_source_triples).astype(np.int32)

    def get_batch_nhop_neighbors_all_context(self, args, batch_sources):
        batch_source_triples_1hop = []
        with open(os.path.join(args, "triple2id_train.txt"), 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                temp = [int(line[0]), int(line[1]), int(line[2])]
                batch_source_triples_1hop.append(temp)
        return np.array(batch_source_triples_1hop).astype(np.int32)

    def gen_subgraph(self, args):
        t1 = time.time()
        file = args.data + "/new_1hop.pickle"
        with open(file, 'rb') as handle:
            new_1hop = pickle.load(handle)
        file = args.data + "/rel_dic.pickle"
        with open(file, 'rb') as handle:
            rel_dic = pickle.load(handle)
        t2 = time.time()
        # print("data_step1_time:{}".format(str(t2 - t1)))
        new_entity2id, new_rel_dic = self.specify_node_update(rel_dic)
        t3 = time.time()
        # print("data_step2_time:{}".format(str(t3 - t2)))
        multi_edge = self.gen_multi_adj(new_rel_dic, new_entity2id)
        t4 = time.time()
        # print("multi_edge_time:{}".format(str(t4 - t3)))
        return new_1hop, new_entity2id, multi_edge

    def get_validation_pred(self, args, model, unique_entities):
        average_hits_at_10000_head, average_hits_at_10000_tail = [], []
        average_hits_at_1000_head, average_hits_at_1000_tail = [], []
        average_hits_at_500_head, average_hits_at_500_tail = [], []
        average_hits_at_100_head, average_hits_at_100_tail = [], []
        average_hits_at_ten_head, average_hits_at_ten_tail = [], []
        average_hits_at_three_head, average_hits_at_three_tail = [], []
        average_hits_at_one_head, average_hits_at_one_tail = [], []
        average_mean_rank_head, average_mean_rank_tail = [], []
        average_mean_recip_rank_head, average_mean_recip_rank_tail = [], []

        for iters in range(1):
            start_time = time.time()
            indices = [i for i in range(len(self.test_indices))]
            batch_indices = self.test_indices[indices, :]
            # print("Sampled indices")
            # print("test set length ", len(self.test_indices))
            entity_list = [j for i, j in self.entity2id.items()]

            ranks_head, ranks_tail = [], []
            reciprocal_ranks_head, reciprocal_ranks_tail = [], []
            hits_at_10000_head, hits_at_10000_tail = 0, 0
            hits_at_1000_head, hits_at_1000_tail = 0, 0
            hits_at_500_head, hits_at_500_tail = 0, 0
            hits_at_100_head, hits_at_100_tail = 0, 0
            hits_at_ten_head, hits_at_ten_tail = 0, 0
            hits_at_three_head, hits_at_three_tail = 0, 0
            hits_at_one_head, hits_at_one_tail = 0, 0

            for i in (range(batch_indices.shape[0])):
                # print(len(ranks_head))
                start_time_it = time.time()
                new_x_batch_head = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))
                new_x_batch_tail = np.tile(
                    batch_indices[i, :], (len(self.entity2id), 1))

                if (batch_indices[i, 0] not in unique_entities or batch_indices[i, 2] not in unique_entities):
                    continue

                new_x_batch_head[:, 0] = entity_list
                new_x_batch_tail[:, 2] = entity_list

                last_index_head = []  # array of already existing triples
                last_index_tail = []
                for tmp_index in range(len(new_x_batch_head)):
                    temp_triple_head = (new_x_batch_head[tmp_index][0], new_x_batch_head[tmp_index][1],
                                        new_x_batch_head[tmp_index][2])
                    if temp_triple_head in self.valid_triples_dict.keys():
                        last_index_head.append(tmp_index)

                    temp_triple_tail = (new_x_batch_tail[tmp_index][0], new_x_batch_tail[tmp_index][1],
                                        new_x_batch_tail[tmp_index][2])
                    if temp_triple_tail in self.valid_triples_dict.keys():
                        last_index_tail.append(tmp_index)

                # Deleting already existing triples, leftover triples are invalid, according
                # to train, validation and test data_pro
                # Note, all of them maynot be actually invalid
                new_x_batch_head = np.delete(
                    new_x_batch_head, last_index_head, axis=0)
                new_x_batch_tail = np.delete(
                    new_x_batch_tail, last_index_tail, axis=0)

                # adding the current valid triples to the Top_Test, i.e, index 0
                new_x_batch_head = np.insert(
                    new_x_batch_head, 0, batch_indices[i], axis=0)
                new_x_batch_tail = np.insert(
                    new_x_batch_tail, 0, batch_indices[i], axis=0)

                # Have to do this, because it doesn't fit in memory

                scores_head = model.batch_test(new_x_batch_head)

                sorted_scores_head, sorted_indices_head = torch.sort(
                    scores_head.view(-1), dim=-1, descending=True)
                # Just search for zeroth index in the sorted scores, we appended valid triple at Top_Test
                ranks_head.append(
                    np.where(sorted_indices_head.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])

                # Tail part here
                scores_tail = model.batch_test(new_x_batch_tail)

                sorted_scores_tail, sorted_indices_tail = torch.sort(
                    scores_tail.view(-1), dim=-1, descending=True)

                # Just search for zeroth index in the sorted scores, we appended valid triple at Top_Test
                ranks_tail.append(
                    np.where(sorted_indices_tail.cpu().numpy() == 0)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                # print("sample - ", ranks_head[-1], ranks_tail[-1])

            for i in range(len(ranks_head)):
                if ranks_head[i] <= 10000:
                    hits_at_10000_head = hits_at_10000_head + 1
                if ranks_head[i] <= 1000:
                    hits_at_1000_head = hits_at_1000_head + 1
                if ranks_head[i] <= 500:
                    hits_at_500_head = hits_at_500_head + 1
                if ranks_head[i] <= 100:
                    hits_at_100_head = hits_at_100_head + 1
                if ranks_head[i] <= 10:
                    hits_at_ten_head = hits_at_ten_head + 1
                if ranks_head[i] <= 3:
                    hits_at_three_head = hits_at_three_head + 1
                if ranks_head[i] == 1:
                    hits_at_one_head = hits_at_one_head + 1

            for i in range(len(ranks_tail)):
                if ranks_tail[i] <= 10000:
                    hits_at_10000_tail = hits_at_10000_tail + 1
                if ranks_tail[i] <= 1000:
                    hits_at_1000_tail = hits_at_1000_tail + 1
                if ranks_tail[i] <= 500:
                    hits_at_500_tail = hits_at_500_tail + 1
                if ranks_tail[i] <= 100:
                    hits_at_100_tail = hits_at_100_tail + 1
                if ranks_tail[i] <= 10:
                    hits_at_ten_tail = hits_at_ten_tail + 1
                if ranks_tail[i] <= 3:
                    hits_at_three_tail = hits_at_three_tail + 1
                if ranks_tail[i] == 1:
                    hits_at_one_tail = hits_at_one_tail + 1

            assert len(ranks_head) == len(reciprocal_ranks_head)
            assert len(ranks_tail) == len(reciprocal_ranks_tail)
            average_hits_at_10000_head.append(
                hits_at_10000_head / len(ranks_head))
            average_hits_at_1000_head.append(
                hits_at_1000_head / len(ranks_head))
            average_hits_at_500_head.append(
                hits_at_500_head / len(ranks_head))
            average_hits_at_100_head.append(
                hits_at_100_head / len(ranks_head))
            average_hits_at_ten_head.append(
                hits_at_ten_head / len(ranks_head))
            average_hits_at_three_head.append(
                hits_at_three_head / len(ranks_head))
            average_hits_at_one_head.append(
                hits_at_one_head / len(ranks_head))
            average_mean_rank_head.append(sum(ranks_head) / len(ranks_head))
            average_mean_recip_rank_head.append(
                sum(reciprocal_ranks_head) / len(reciprocal_ranks_head))

            average_hits_at_10000_tail.append(
                hits_at_10000_tail / len(ranks_head))
            average_hits_at_1000_tail.append(
                hits_at_1000_tail / len(ranks_head))
            average_hits_at_500_tail.append(
                hits_at_500_tail / len(ranks_head))
            average_hits_at_100_tail.append(
                hits_at_100_tail / len(ranks_head))
            average_hits_at_ten_tail.append(
                hits_at_ten_tail / len(ranks_head))
            average_hits_at_three_tail.append(
                hits_at_three_tail / len(ranks_head))
            average_hits_at_one_tail.append(
                hits_at_one_tail / len(ranks_head))
            average_mean_rank_tail.append(sum(ranks_tail) / len(ranks_tail))
            average_mean_recip_rank_tail.append(
                sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail))

        cumulative_hits_10000 = (sum(average_hits_at_10000_head) / len(average_hits_at_10000_head)
                                 + sum(average_hits_at_10000_tail) / len(average_hits_at_10000_tail)) / 2
        cumulative_hits_1000 = (sum(average_hits_at_1000_head) / len(average_hits_at_1000_head)
                                + sum(average_hits_at_1000_tail) / len(average_hits_at_1000_tail)) / 2
        cumulative_hits_500 = (sum(average_hits_at_500_head) / len(average_hits_at_500_head)
                               + sum(average_hits_at_500_tail) / len(average_hits_at_500_tail)) / 2
        cumulative_hits_100 = (sum(average_hits_at_100_head) / len(average_hits_at_100_head)
                               + sum(average_hits_at_100_tail) / len(average_hits_at_100_tail)) / 2
        cumulative_hits_ten = (sum(average_hits_at_ten_head) / len(average_hits_at_ten_head)
                               + sum(average_hits_at_ten_tail) / len(average_hits_at_ten_tail)) / 2
        cumulative_hits_three = (sum(average_hits_at_three_head) / len(average_hits_at_three_head)
                                 + sum(average_hits_at_three_tail) / len(average_hits_at_three_tail)) / 2
        cumulative_hits_one = (sum(average_hits_at_one_head) / len(average_hits_at_one_head)
                               + sum(average_hits_at_one_tail) / len(average_hits_at_one_tail)) / 2
        cumulative_mean_rank = (sum(average_mean_rank_head) / len(average_mean_rank_head)
                                + sum(average_mean_rank_tail) / len(average_mean_rank_tail)) / 2
        cumulative_mean_recip_rank = (sum(average_mean_recip_rank_head) / len(average_mean_recip_rank_head) + sum(
            average_mean_recip_rank_tail) / len(average_mean_recip_rank_tail)) / 2

        print("\nCumulative stats are -> ")
        # print("Hits@10000 are {}".format(cumulative_hits_10000))
        # print("Hits@1000 are {}".format(cumulative_hits_1000))
        # print("Hits@500 are {}".format(cumulative_hits_500))
        # print("Hits@100 are {}".format(cumulative_hits_100))
        # print("Hits@10 are {}".format(cumulative_hits_ten))
        # print("Hits@3 are {}".format(cumulative_hits_three))
        print("Hits@1 are {}".format(cumulative_hits_one))
        print("Mean rank {}".format(cumulative_mean_rank))
        # print("Mean Reciprocal Rank {}".format(cumulative_mean_recip_rank))
        return cumulative_hits_10000, cumulative_hits_1000, cumulative_hits_500, \
               cumulative_hits_100, cumulative_hits_ten, cumulative_hits_three, \
               cumulative_hits_one, cumulative_mean_rank, cumulative_mean_recip_rank




