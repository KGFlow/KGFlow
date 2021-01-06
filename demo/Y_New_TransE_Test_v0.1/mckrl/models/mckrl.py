import torch
import time
import random
import math
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mckrl.models.gcn_encoder import GCN_Encoder
from mckrl.models.attention import Attention_out
from mckrl.models.spgat import SpGAT
# from mckrl.config.config import *
from mckrl.data_process.creat_rel_generator import next_batch
from mckrl.models.dgi import DGI
import pickle
from tqdm import tqdm
import tf_geometric as tfg
import scipy.sparse as sp
from torch_scatter import scatter_max, scatter_sum, scatter_mean, scatter_softmax, scatter_add, scatter_min


def norm_embeddings(embeddings: torch.Tensor):
    norm = embeddings.norm(p=2, dim=-1, keepdim=True)
    return embeddings / norm

class MCKRL(nn.Module):
    def __init__(self, args, nfeat, nhid1, nhid2, dropout, initial_entity_emb,
                 initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, ft_size, hid_units, nonlinearity, Corpus_):
        super(MCKRL, self).__init__()
        self.SGCN1 = GCN_Encoder(200, nhid1, nhid2, dropout)
        self.dropout = 0.1
        self.attention_out = Attention_out(200, args.attention_hidden_size)
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]
        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]
        self.drop_GAT = drop_GAT
        self.alpha = alpha  # For leaky relu
        self.sparse_gat_1 = SpGAT(self.num_nodes, 100, self.entity_out_dim_1, 100,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1)
        self.layer_relation = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(self.dropout)

        )
        self.layer_relation_gat = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout)

        )
        self.layer_relation_img = nn.Sequential(
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(self.dropout)

        )

        self.layer_relation_input = nn.Sequential(
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(self.dropout)

        )

        self.W_entities = nn.Parameter(torch.zeros(
            size=(100, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(initial_entity_emb.shape[0], 200))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(initial_relation_emb.shape[0], 200))

        self.W_relations_out = nn.Parameter(torch.zeros(
            size=(400, 200)))
        nn.init.xavier_uniform_(self.W_relations_out.data, gain=1.414)
        self.DGI = DGI(ft_size, hid_units, nonlinearity)
        # input-entity-multi-modal-feature
        self.img_feat = (initial_entity_emb[:, :4096])
        self.text_feat = (initial_entity_emb[:, 4096:])
        # input-relation-multi-modal-feature
        self.relation_embeddings = initial_relation_emb
        self.relation_embeddings_ = nn.Parameter(torch.zeros(
            size=(initial_relation_emb.shape[0], 100)))
        nn.init.xavier_uniform_(self.relation_embeddings_.data, gain=1.414)
        self.entity_embeddings_ = nn.Parameter(torch.zeros(
            size=(initial_entity_emb.shape[0], 100)))
        nn.init.xavier_uniform_(self.entity_embeddings_.data, gain=1.414)

        self.img_encoder = nn.Sequential(
            nn.Linear(4096, 300),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.img_text_encoder = nn.Sequential(
            nn.Linear(4396, 100),
            nn.Dropout(self.dropout)
        )
        self.edge_index = self.read_edge_index(args)
        self.layer_emb = nn.Sequential(
            nn.Linear(400, 1),
        )
        self.layer_emb_out = nn.Sequential(
            nn.Linear(400, 200),
            nn.Dropout(self.dropout)

        )
        self.b_x, self.b_node_graph_index, self.b_edge_index, self.b_new_adj = self.batch_graph_gen(
            Corpus_.new_entity2id)

    def edge_index_gen(self, new_entity2id, new_1hop):
        edge_index = []
        count = 0
        for key, val in tqdm(new_entity2id.items()):
            print(count)
            count += 1
            ori = []
            dst = []
            entity_new_id = []
            entity_ori_id = []
            for tri_key, tri_val in val.items():
                entity_new_id.append(tri_val)
                entity_ori_id.append(tri_key)
            if len(entity_new_id) == 2:
                edge_index.append(np.array([[entity_new_id[0]], [entity_new_id[1]]]))
            else:
                temp_tris = []
                for key_, val_ in new_1hop.items():
                    if key_ in entity_ori_id:
                        for tri in val_:
                            if tri[1] == key and tri not in temp_tris:
                                temp_tris.append(tri)
                for tri in temp_tris:
                    ori.append(val[tri[0]])
                    dst.append(val[tri[2]])
                edge_index.append(np.array([ori, dst]))
        file = "edge_index3.pickle"
        with open(file, 'wb') as handle:
            pickle.dump(edge_index, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def read_edge_index(self, args):
        file = args.data + "edge_index3.pickle"
        with open(file, 'rb') as handle:
            edge_index = pickle.load(handle)
        return edge_index

    def big_graph_edges_gen(self, batch_graph):
        ori = batch_graph.edge_index.numpy()[0]
        dst = batch_graph.edge_index.numpy()[1]
        l = len(ori)
        with open("big_graph_edges.txt", 'a', encoding="utf-8") as f:
            for i in range(l):
                temp = str(ori[i]) + "\t" + str(dst[i]) + "\n"
                f.write(temp)

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

    def gen_adj(self, sample_struct_edges):
        shape_size = max([max(sample_struct_edges[:, 0]), max(sample_struct_edges[:, 1])]) + 1
        sedges = np.array(list(sample_struct_edges), dtype=np.int32).reshape(sample_struct_edges.shape)
        sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                             shape=(shape_size, shape_size),
                             dtype=np.float32)
        sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
        nsadj = self.normalize(sadj + sp.eye(sadj.shape[0]))
        nsadj = self.sparse_mx_to_torch_sparse_tensor(nsadj)
        return nsadj

    def batch_graph_gen(self, new_entity2id):
        big_graph = []
        entity_index = []
        for key, val in new_entity2id.items():
            temp = []
            for k in val.keys():
                temp.append(k)
            entity_index.append(temp)
        for i in range(len(new_entity2id)):
            sub_graph = tfg.Graph(x=entity_index[i], edge_index=self.edge_index[i])
            big_graph.append(sub_graph)
        batch_graph = tfg.BatchGraph.from_graphs(big_graph)
        new_edges = []
        edges_data = batch_graph.edge_index.numpy()
        for i in range(len(edges_data[0])):
            new_edges.append([edges_data[0][i], edges_data[1][i]])
        new_adj = self.gen_adj(np.array(np.array(new_edges)))
        return (batch_graph.x.numpy()), (batch_graph.node_graph_index.numpy()), (batch_graph.edge_index.numpy()), (
            new_adj)

    def multi_context_encoder(self, entity_multi_modal_feat):
        rel_embed_context = self.layer_relation_input(self.relation_embeddings)
        new_entity_rel_embed = torch.cat(
            [entity_multi_modal_feat[self.b_x], rel_embed_context[self.b_node_graph_index]], dim=1)
        entity_embed = self.SGCN1(new_entity_rel_embed.cuda(), self.b_new_adj.cuda())
        index = torch.tensor(self.b_x).long().cuda()
        out = scatter_mean(entity_embed, index, dim=0)
        z = out[index]
        emb = torch.cat([entity_embed, z], dim=-1)
        new_emb = self.layer_emb(emb)
        z_s = scatter_softmax(new_emb, index, dim=0)
        new_out = scatter_add(z_s * emb, index, dim=0)
        new_out = self.layer_emb_out(new_out)
        rel_context = self.layer_relation(self.relation_embeddings)
        return new_out, rel_context

    def multi_topology_encoder(self, Corpus_, adj, batch_inputs, train_indices_nhop, entity_multi_modal_feat):
        # multi-topology encoder
        # getting edge list
        rel_embed = self.layer_relation_gat(self.relation_embeddings)
        edge_list = adj[0]
        edge_type = adj[1]
        edge_list_nhop = torch.cat(
            (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
        edge_type_nhop = torch.cat(
            [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        if (CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()

        edge_embed = rel_embed[edge_type]

        entity_multi_modal_feat = F.normalize(
            entity_multi_modal_feat, p=2, dim=1).detach()

        # rel_embed = F.normalize(
        #     rel_embed, p=2, dim=1).detach()

        out_entity_1, out_relation_top = self.sparse_gat_1(
            Corpus_, batch_inputs, entity_multi_modal_feat, rel_embed,
            edge_list, edge_type, edge_embed, edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(entity_multi_modal_feat.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = entity_multi_modal_feat.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
                       mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1
        out_entity_top = F.normalize(out_entity_1, p=2, dim=1)
        return out_entity_top, out_relation_top

    def gen_shuf_fts(self, entity):
        entity_ = entity[np.newaxis]
        idx = np.random.permutation(len(entity))
        # add Guass noise
        shuf_fts = entity_[:, idx, :] + torch.randn(entity_.shape).cuda()
        return entity_, shuf_fts

    def forward(self, Corpus_, adj, batch_inputs, train_indices_nhop, sp_adj, sparse):
        # 对输入进行预处理
        # img_feat = self.img_encoder(self.img_feat.cuda())
        # text_feat = self.text_encoder(self.text_feat.cuda())
        entity_multi_modal_feat = self.img_text_encoder(torch.cat([self.img_feat.cuda(), self.text_feat.cuda()], dim=1))

        rel_embed = self.layer_relation_gat(self.relation_embeddings.cuda())
        # #文本-图像-变量参数
        new_entity_embed = (torch.cat([entity_multi_modal_feat, self.entity_embeddings_], dim=1))
        new_entity_embed = norm_embeddings(new_entity_embed)
        #
        new_rel_embed = (torch.cat([rel_embed, self.relation_embeddings_], dim=1))
        new_rel_embed = norm_embeddings(new_rel_embed)

        self.final_entity_embeddings.data = new_entity_embed.data
        self.final_relation_embeddings.data = new_rel_embed.data

        return new_entity_embed, new_rel_embed

