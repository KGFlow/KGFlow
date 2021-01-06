import torch
import torch.nn as nn
import torch.nn.functional as F
from mckrl.layers.layer import GraphConvolution


class GCN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN_Encoder, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = 0.2

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
