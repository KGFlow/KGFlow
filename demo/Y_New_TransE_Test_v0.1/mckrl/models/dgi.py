import torch
import torch.nn as nn
from mckrl.layers.gcn_dgi import GCN
from mckrl.layers.readout import AvgReadout
from mckrl.layers.discriminator import Discriminator


class DGI(nn.Module):
    def __init__(self, n_in, n_h, activation):
        super(DGI, self).__init__()
        self.gcn = GCN(n_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()
        self.dropout = 0.1
        self.disc = Discriminator(n_h)
        self.MLP_CL1 = nn.Sequential(
            nn.Linear(200, 400),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(400, 200)
        )

    def forward(self, seq1_top, seq2_top, seq1_con, seq2_con, adj, sparse, msk, samp_bias1, samp_bias2):
        # topology
        # h_1_top_node = self.gcn(seq1_top, adj.cuda(), sparse)
        h_1_top_node = seq1_top
        # h_1_top_node_ = self.MLP_CL1(h_1_top_node)

        c_top_graph = self.read(h_1_top_node, msk)
        c_top_graph = self.sigm(c_top_graph)
        # h_2_top_node = self.gcn(seq2_top, adj.cuda(), sparse)
        h_2_top_node = seq2_top
        # h_2_top_node_ = self.MLP_CL1(h_2_top_node)

        # context
        # h_1_con_node = self.gcn(seq1_con, adj.cuda(), sparse)
        h_1_con_node = seq1_con
        # h_1_con_node_ = self.MLP_CL1(h_1_con_node)

        c_con_graph = self.read(h_1_con_node, msk)
        c_con_graph = self.sigm(c_con_graph)
        # h_2_con_node = self.gcn(seq2_con, adj.cuda(), sparse)
        h_2_con_node = seq2_con
        # h_2_con_node_ = self.MLP_CL1(h_2_con_node)

        ret = self.disc(c_top_graph, h_1_con_node, h_2_con_node, samp_bias1, samp_bias2) + \
              self.disc(c_con_graph, h_1_top_node, h_2_top_node, samp_bias1, samp_bias2)

        return ret
