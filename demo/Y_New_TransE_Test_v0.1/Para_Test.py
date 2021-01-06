# coding = utf-8
from torch.autograd import Variable
from mckrl.models.decoder import Decoder
from mckrl.data_process.preprocess import load_graph_dgi
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mckrl.models.gcn_encoder import GCN_Encoder
from mckrl.models.attention import Attention_out
from mckrl.models.spgat import SpGAT
from mckrl.models.dgi import DGI
import pickle
from mckrl.config.config import *
from tqdm import tqdm
import tf_geometric as tfg
import scipy.sparse as sp
# from torch_scatter import scatter_max, scatter_sum, scatter_mean, scatter_softmax, scatter_add, scatter_min
import argparse, sys
from mckrl.utils.func import batch_gat_loss, load_data, save_model, norm_embeddings
from mckrl.models.convkb import ConvKB
import torch
from tensorboardX import SummaryWriter

torch.manual_seed(1234)


# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
def parse_args():
    args = argparse.ArgumentParser()
    # network arguments
    args.add_argument("-data", "--data",
                      default="./data/DB15K/", help="data directory")
    args.add_argument("-e_g", "--epochs_gat", type=int,
                      default=3600, help="Number of epochs")
    args.add_argument("-e_c", "--epochs_conv", type=int,
                      default=200, help="Number of epochs")
    args.add_argument("-w_gat", "--weight_decay_gat", type=float,
                      default=5e-6, help="L2 reglarization for gat")
    # args.add_argument("-w_conv", "--weight_decay_conv", type=float,
    #                   default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-w_conv", "--weight_decay_conv", type=float,
                      default=1e-5, help="L2 reglarization for conv")
    args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
    args.add_argument("-emb_size", "--embedding_size", type=int,
                      default=100, help="Size of embeddings (if pretrained not used)")
    args.add_argument("-l", "--lr", type=float, default=1e-3)
    args.add_argument("-g2hop", "--get_2hop", type=bool, default=True)
    args.add_argument("-u2hop", "--use_2hop", type=bool, default=True)
    args.add_argument("-p2hop", "--partial_2hop", type=bool, default=True)
    args.add_argument("-outfolder", "--output_folder",
                      default="./checkpoints/db/out/", help="Folder name to save the models.")

    # arguments for GAT
    args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")
    args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
    args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layers")
    args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layers")
    args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
    args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
    args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")

    # arguments for convolution network
    args.add_argument("-b_conv", "--batch_size_conv", type=int,
                      default=128, help="Batch size for conv")
    args.add_argument("-alpha_conv", "--alpha_conv", type=float,
                      default=0.2, help="LeakyRelu alphas for conv layers")
    args.add_argument("-neg_s_conv", "--valid_invalid_ratio_conv", type=int, default=40,
                      help="Ratio of valid to invalid triples for convolution training")
    args.add_argument("-o", "--out_channels", type=int, default=50,
                      help="Number of output channels in conv layers")
    args.add_argument("-drop_conv", "--drop_conv", type=float,
                      default=0.0, help="Dropout probability for convolution layers")

    # Single Attention
    args.add_argument("-attention_hidden_size", "--attention_hidden_size", type=int,
                      default=16, help="Attention_hiden_size for attention layers")

    args.add_argument("-model_name", "--model_name", type=str,
                      default="Con_Test")

    cmd = "--data ./data/DB15K/  " \
          "--model_name Para_Test " \
          "--epochs_gat 20001 " \
          "--valid_invalid_ratio_gat 2 " \
          "--epochs_conv 301 " \
          "--weight_decay_gat 0.00001 " \
          "--batch_size_gat 10000 " \
          "--margin 1 " \
          "--output_folder ./checkpoints/db/con_/ " \
          "--out_channels 50 " \
          "--attention_hidden_size 128 "

    sys.argv += cmd.split()
    args = args.parse_args()
    return args


class MCKRL(nn.Module):
    def __init__(self, args, nfeat, nhid1, nhid2, dropout, initial_entity_emb,
                 initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, ft_size, hid_units, nonlinearity, Corpus_):
        super(MCKRL, self).__init__()

        self.dropout = dropout

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

        # self.b_x, self.b_node_graph_index, self.b_edge_index, self.b_new_adj = self.batch_graph_gen(
        #     Corpus_.new_entity2id)
        self.convKB = ConvKB(100, 3, 1, 50, 0, 0.2)

    def forward(self, Corpus_, batch_inputs):
        # 对输入进行预处理
        new_entity_embed = self.entity_embeddings_
        new_entity_embed = norm_embeddings(new_entity_embed)
        new_rel_embed = self.relation_embeddings_

        new_rel_embed = norm_embeddings(new_rel_embed)
        new_entity_embed = norm_embeddings(new_entity_embed)
        new_rel_embed = norm_embeddings(new_rel_embed)

        self.final_entity_embeddings.data = new_entity_embed.data
        self.final_relation_embeddings.data = new_rel_embed.data

        conv_input = torch.cat((new_entity_embed[batch_inputs[:, 0], :].unsqueeze(1),
                                new_rel_embed[batch_inputs[:, 1]].unsqueeze(1),
                                new_entity_embed[batch_inputs[:, 2], :].unsqueeze(1)),
                               dim=1)
        out_conv = self.convKB(conv_input)

        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat((self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1),
                                self.final_relation_embeddings[batch_inputs[:, 1]].unsqueeze(1),
                                self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
                               dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv


args = parse_args()

Corpus_, entity_embeddings, relation_embeddings = load_data(args)


def train_encoder(args):
    t1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))

    model_encoder = MCKRL(args, nfeat=200, nhid1=400, nhid2=200, dropout=0.2,
                          initial_entity_emb=entity_embeddings, initial_relation_emb=relation_embeddings,
                          entity_out_dim=args.entity_out_dim, relation_out_dim=args.entity_out_dim,
                          drop_GAT=args.drop_GAT, alpha=args.alpha, nheads_GAT=args.nheads_GAT, ft_size=200,
                          hid_units=512, nonlinearity='prelu', Corpus_=Corpus_,
                          )
    if CUDA:  model_encoder.cuda()
    optimizer = torch.optim.Adam(model_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay_gat)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5, last_epoch=-1)
    gat_loss_func = nn.MarginRankingLoss(margin=args.margin)
    margin_loss = torch.nn.SoftMarginLoss()
    b_xent = nn.BCEWithLogitsLoss()

    epoch_losses = []  # losses of all epochs
    print("Number of epochs {}".format(args.epochs_gat))
    lbl = torch.cat((torch.ones(1, len(entity_embeddings)), torch.zeros(1, len(entity_embeddings))), 1)
    with open("./train_log/{}/{}_log.txt".format(args.model_name, args.model_name), "a",
              encoding="utf-8")as log_encoder:
        for epoch in tqdm(range(args.epochs_gat)):
            print("\nepoch-> ", epoch)
            random.shuffle(Corpus_.train_triples)
            Corpus_.train_indices = np.array(
                list(Corpus_.train_triples)).astype(np.int32)
            model_encoder.train()  # getting in training mode
            start_time = time.time()
            epoch_loss = []
            if len(Corpus_.train_indices) % args.batch_size_gat == 0:
                num_iters_per_epoch = len(Corpus_.train_indices) // args.batch_size_gat
            else:
                num_iters_per_epoch = (len(Corpus_.train_indices) // args.batch_size_gat) + 1
            for iters in (range(num_iters_per_epoch)):
                start_time_iter = time.time()
                train_indices, train_values = Corpus_.get_iteration_batch(iters)
                if CUDA:
                    train_indices = Variable(torch.LongTensor(train_indices)).cuda()
                    train_values = Variable(torch.FloatTensor(train_values)).cuda()
                    lbl = lbl.cuda()
                else:
                    train_indices = Variable(torch.LongTensor(train_indices))
                    train_values = Variable(torch.FloatTensor(train_values))
                # forward pass
                preds = model_encoder(Corpus_,

                                      train_indices,

                                      )
                optimizer.zero_grad()
                tri_loss = margin_loss(preds.view(-1), train_values.view(-1))

                loss = tri_loss

                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.data.item())
            scheduler.step()
            print("Epoch {} , average loss {} , epoch_time {}".format(
                epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
            epoch_losses.append(sum(epoch_loss) / len(epoch_loss))

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
            temp = "Epoch: {}\ntri_loss: {} average loss: {}\nepoch_time: {} current_time:{} \n".format(epoch,
                                                                                                        tri_loss,
                                                                                                        sum(
                                                                                                            epoch_loss) / len(
                                                                                                            epoch_loss),
                                                                                                        time.time() - start_time,
                                                                                                        current_time)
            log_encoder.write(temp)

            # if epoch % 100 == 0:
            #     save_model(model_encoder, epoch, args.output_folder)
            if (epoch + 1) % 200 == 0:
                model_encoder.eval()
                with torch.no_grad():
                    hits_10000, hits_1000, hits_500, \
                    hits_100, hits_10, hits_3, \
                    hits_1, mean_rank, mean_recip_rank = Corpus_.get_validation_pred(args, model_encoder,
                                                                                     Corpus_.unique_entities_train)
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
                    # with open("./results/train_log/{}/log_{}.txt".format(args.model_name, args.model_name), "a",
                    #           encoding="utf-8") as results_log:
                    temp = "Current_time: {}\n" \
                           "hits_10000:  {}\n" \
                           "hits_1000:  {}\n" \
                           "hits_500:  {}\n" \
                           "hits_100:  {}\n" \
                           "hits_10:  {}\n" \
                           "hits_3:  {}\n" \
                           "hits_1:  {}\n" \
                           "mean_rank:  {}\n" \
                           "mean_recip_rank:  {}\n".format(str(current_time), hits_10000, hits_1000,
                                                           hits_500,
                                                           hits_100, hits_10, hits_3,
                                                           hits_1, mean_rank, mean_recip_rank)
                    log_encoder.write(temp)


def evaluate_test(args, unique_entities):
    model_encoder = MCKRL(args, nfeat=200, nhid1=400, nhid2=200, dropout=0.2,
                          initial_entity_emb=entity_embeddings, initial_relation_emb=relation_embeddings,
                          entity_out_dim=args.entity_out_dim, relation_out_dim=args.entity_out_dim,
                          drop_GAT=args.drop_GAT, alpha=args.alpha, nheads_GAT=args.nheads_GAT, ft_size=200,
                          hid_units=512, nonlinearity='prelu', Corpus_=Corpus_,
                          )
    model_encoder.load_state_dict(torch.load(
        '{0}/trained_{1}.pth'.format(args.output_folder, 5000)), strict=False)

    model_encoder.cuda()
    model_encoder.eval()
    with torch.no_grad():
        Corpus_.get_validation_pred(args, model_encoder, unique_entities)


train_encoder(args)

# evaluate_test(args, Corpus_.unique_entities_train)
