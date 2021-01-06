import torch
import os
import numpy as np
import pickle
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx


def read_entity_from_id(filename='./data_pro/FB15k-237/entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id


def read_relation_from_id(filename='./data_pro/FB15k-237/relation2id.txt'):
    relation2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                relation, relation_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                relation2id[relation] = int(relation_id)
    return relation2id


def init_embeddings(entity_file, relation_file):
    with open(entity_file, 'rb') as handle:
        entity_emb = pickle.load(handle)
    with open(relation_file, 'rb') as handle:
        relation_emb = pickle.load(handle)
    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)


# def init_embeddings(entity_file, relation_file):
#     entity_emb, relation_emb = [], []
#     with open(entity_file) as f:
#         for line in f:
#             entity_emb.append([float(val) for val in line.strip().split()])
#
#     with open(relation_file) as f:
#         for line in f:
#             relation_emb.append([float(val) for val in line.strip().split()])
#
#     return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 = line[0].strip(), line[1].strip(), line[2].strip()
    return e1, relation, e2


def load_data(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()

    # this is list for relation triples
    triples_data = []

    # for sparse tensor, rows list contains corresponding row of sparse tensor, cols list contains corresponding
    # columnn of sparse tensor, data_pro contains the type of relation
    # Adjacecny matrix of entities is undirected, as the source and tail entities should know, the relation
    # type they are connected with
    rows, cols, data = [], [], []
    unique_entities = set()
    for line in lines:
        e1, relation, e2 = parse_line(line)
        unique_entities.add(e1)
        unique_entities.add(e2)
        triples_data.append(
            (entity2id[e1], relation2id[relation], entity2id[e2]))
        if not directed:
            # Connecting source and tail entity
            rows.append(entity2id[e1])
            cols.append(entity2id[e2])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])

        # Connecting tail and source entity
        rows.append(entity2id[e2])
        cols.append(entity2id[e1])
        if is_unweigted:
            data.append(1)
        else:
            data.append(relation2id[relation])

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def load_data_test(filename, entity2id, relation2id, is_unweigted=False, directed=True):
    with open(filename) as f:
        lines = f.readlines()
    triples_data = []
    rows, cols, data = [], [], []
    unique_entities = set()
    # 打乱
    # idx = np.random.permutation(7416)[:500]
    # 不打乱
    idx = [i for i in range(100)]
    for i, line in enumerate(lines):
        if i < 100:
            e1, relation, e2 = parse_line(line)
            unique_entities.add(e1)
            unique_entities.add(e2)
            triples_data.append(
                (entity2id[e1], relation2id[relation], entity2id[e2]))
            if not directed:
                # Connecting source and tail entity
                rows.append(entity2id[e1])
                cols.append(entity2id[e2])
                if is_unweigted:
                    data.append(1)
                else:
                    data.append(relation2id[relation])

            # Connecting tail and source entity
            rows.append(entity2id[e2])
            cols.append(entity2id[e1])
            if is_unweigted:
                data.append(1)
            else:
                data.append(relation2id[relation])
        else:
            break

    print("number of unique_entities ->", len(unique_entities))
    return triples_data, (rows, cols, data), list(unique_entities)


def build_data(path='./data_pro/FB15k-237/', is_unweigted=False, directed=True):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    relation2id = read_relation_from_id(path + 'relation2id.txt')

    # Adjacency matrix only required for training phase
    # Currenlty creating as unweighted, undirected
    train_triples, train_adjacency_mat, unique_entities_train = load_data(os.path.join(
        path, 'train.txt'), entity2id, relation2id, is_unweigted, directed)
    validation_triples, valid_adjacency_mat, unique_entities_validation = load_data(
        os.path.join(path, 'valid.txt'), entity2id, relation2id, is_unweigted, directed)
    test_triples, test_adjacency_mat, unique_entities_test = load_data(os.path.join(
        path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)
    # 使用小测试集
    # print("*****\n test_l \n******")
    # test_triples, test_adjacency_mat, unique_entities_test = load_data_test(os.path.join(
    #     path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    left_entity, right_entity = {}, {}

    with open(os.path.join(path, 'train.txt')) as f:
        lines = f.readlines()

    for line in lines:
        e1, relation, e2 = parse_line(line)

        # Count number of occurences for each (e1, relation)
        if relation2id[relation] not in left_entity:
            left_entity[relation2id[relation]] = {}
        if entity2id[e1] not in left_entity[relation2id[relation]]:
            left_entity[relation2id[relation]][entity2id[e1]] = 0
        left_entity[relation2id[relation]][entity2id[e1]] += 1

        # Count number of occurences for each (relation, e2)
        if relation2id[relation] not in right_entity:
            right_entity[relation2id[relation]] = {}
        if entity2id[e2] not in right_entity[relation2id[relation]]:
            right_entity[relation2id[relation]][entity2id[e2]] = 0
        right_entity[relation2id[relation]][entity2id[e2]] += 1

    left_entity_avg = {}
    for i in range(len(relation2id)):
        left_entity_avg[i] = sum(
            left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_entity_avg = {}
    for i in range(len(relation2id)):
        right_entity_avg[i] = sum(
            right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = 1000 * right_entity_avg[i] / \
                              (right_entity_avg[i] + left_entity_avg[i])

    return (train_triples, train_adjacency_mat), (validation_triples, valid_adjacency_mat), (
    test_triples, test_adjacency_mat), \
           entity2id, relation2id, headTailSelector, unique_entities_train


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_graph_dgi(config):
    struct_edges = np.genfromtxt(config.data + "edges_train.txt", dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(12842, 12842),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj + sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return nsadj
