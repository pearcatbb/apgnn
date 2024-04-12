# -*- coding: utf-8 -*-
# @Time     : 2022/11/5 14:45
# @Author   : Guo Jiayu
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from operator import itemgetter
import math



class InterAmazon(nn.Module):

    def __init__(self, features, feature_dim, embed_dim,
                    adj_lists, homo, cuda=True):
        super(InterAmazon, self).__init__()

        self.features = features
        self.dropout = 0.5
        self.adj_lists = adj_lists
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.sap = SimilarityAugment(self.features, self.feat_dim, self.embed_dim, self.adj_lists, self.cuda)

        self.sage1 = GraphSage(2, Encoder(features, feature_dim, embed_dim, homo, MeanAggregator(features, cuda=cuda), gcn=False, cuda=cuda))
        # self.sage2 = GraphSage(2, Encoder(features, feature_dim, embed_dim, adj_lists[1], MeanAggregator(features, cuda=cuda), gcn=False, cuda=cuda))
        # self.sage3 = GraphSage(2, Encoder(features, feature_dim, embed_dim, adj_lists[2], MeanAggregator(features, cuda=cuda), gcn=False, cuda=cuda))
        self.cuda = cuda
        self.hgnn = HGNN_conv(self.feat_dim, self.embed_dim)

        self.old_trans = nn.Sequential(nn.Linear(self.embed_dim + self.feat_dim, self.embed_dim), nn.LeakyReLU())

        # label predictor for similarity measure
        self.label_clf = nn.Linear(self.feat_dim, self.embed_dim)
        self.attention_weight = Parameter(torch.FloatTensor(1, 3))
        init.xavier_uniform_(self.attention_weight)
        self.sap = SimilarityAugment(self.features, self.feat_dim, self.embed_dim, self.adj_lists, self.cuda)

    def forward(self, nodes, labels, train_flag=True):
        # extract 1-hop neighbor ids from adj lists of each single-relation graph
        to_neighs = []
        for adj_list in self.adj_lists:
            if adj_list == None: break
            to_neighs.append([set(adj_list[int(node)]) for node in nodes])

        # find unique nodes and their neighbors used in current batch
        try:
            unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2], set(nodes)))
        except:
            print('')
        # unique_nodes = set.union(set.union(*to_neighs[0]), set(nodes))


        # calculate label-aware scores
        if self.cuda:
            batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
        else:
            batch_features = self.features(torch.LongTensor(list(unique_nodes)))
        batch_scores = self.label_clf(batch_features)
        id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

        # the label-aware scores for current batch of nodes
        center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

        # sap_nodes_embed = self.sap(nodes, batch_scores, id_mapping, to_neighs)

        # embed = torch.cat((center_scores, sap_nodes_embed), dim=1)

        center_X1 = self.sage1(nodes)
        # center_X2 = self.sage2(nodes)
        # center_X3 = self.sage3(nodes)
        # cat_feats_o = torch.cat((center_X1, center_X2, center_X3), dim=1).view(len(nodes), 3, self.embed_dim)
        # weight = F.softmax(self.attention_weight, dim=1)
        # weight = weight.repeat(len(nodes), 1).view((len(nodes), 1, 3))

        # feats_o = torch.bmm(weight, cat_feats_o)
        # return embed
        return center_scores, None, center_X1.t()

class SimilarityAugment(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_lists, is_cuda):
        super(SimilarityAugment, self).__init__()
        self.features = features
        self.feat_dim = feature_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.is_cuda = is_cuda

        self.sfr1 = SimilarityForRelation(self.features, self.feat_dim, self.embed_dim, self.adj_lists[0], self.is_cuda)
        self.sfr2 = SimilarityForRelation(self.features, self.feat_dim, self.embed_dim, self.adj_lists[1], self.is_cuda)
        self.sfr3 = SimilarityForRelation(self.features, self.feat_dim, self.embed_dim, self.adj_lists[2], self.is_cuda)

        self.relations_atten = nn.Parameter(torch.FloatTensor(1, 3))

        # 参数初始化
        init.xavier_uniform_(self.relations_atten)

    def forward(self, nodes, batch_trans_features, id_mapping, to_neighs):

        nodes_embed_r1 = self.sfr1(nodes, to_neighs[0], id_mapping, batch_trans_features)
        nodes_embed_r2 = self.sfr2(nodes, to_neighs[1], id_mapping, batch_trans_features)
        nodes_embed_r3 = self.sfr3(nodes, to_neighs[2], id_mapping, batch_trans_features)

        r_atten_weights = F.softmax(self.relations_atten, dim=0)

        nodes_embed = nodes_embed_r1 * r_atten_weights[0][0] + nodes_embed_r2 * r_atten_weights[0][1] + nodes_embed_r3 * r_atten_weights[0][2]
        # nodes_embed = nodes_embed_r1
        return nodes_embed

class SimilarityForRelation(nn.Module):
    def __init__(self, features, feature_dim, embed_dim, adj_lists, is_cuda):
        super(SimilarityForRelation, self).__init__()
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.is_cuda = is_cuda
        self.features = features
        self.adj_lists = adj_lists

        self.w_ho = nn.Parameter(torch.FloatTensor(2*self.embed_dim, 1))
        self.w_he = nn.Parameter(torch.FloatTensor(2*self.embed_dim, 1))

        self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.embed_dim))
        self.hgnn = HGNN_conv(self.feat_dim, self.embed_dim)
        # 参数初始化
        init.xavier_uniform_(self.w_ho)
        init.xavier_uniform_(self.w_he)
        init.xavier_uniform_(self.weight)



    def forward(self, nodes, to_neigh, id_mapping, batch_trans_features):
        unique_nodes_list = set.union(set.union(*to_neigh), set(nodes))

        # 通过矩阵index找原来的id
        id2nodes_mapping = {i:  n for i,n in enumerate(unique_nodes_list)}
        nodes2id_mapping = {n: i for i, n in enumerate(unique_nodes_list)}

        similarity_vec = batch_trans_features[itemgetter(*unique_nodes_list)(id_mapping), :].view(-1, self.embed_dim)

        similarity_matrix = similarity_vec.mm(similarity_vec.t())
        # 去掉对角线再topK
        self_similarity = torch.eye(similarity_matrix.size()[0], similarity_matrix.size()[0])
        if self.is_cuda:
            self_similarity = self_similarity.cuda()

        similarity_matrix = similarity_matrix - self_similarity
        values, indices = torch.topk(similarity_matrix, 2)

        nodes_indices = [nodes2id_mapping[node] for node in nodes]
        selected_topk_indices = indices[nodes_indices].squeeze().cpu().data.numpy().tolist()

        samp_neighs = [list(neigh.union([id2nodes_mapping[index[0]], id2nodes_mapping[index[1]]]).union({nodes[i]})) for i, (neigh, index) in enumerate(zip(to_neigh, selected_topk_indices))]

        neighs = [token for st in samp_neighs for token in st]
        unique_nodes_list = list(set.union(set(neighs)))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        G_, H_ = construct_G(unique_nodes, samp_neighs, cuda=self.is_cuda)
        if self.cuda:
            # self_feats = self.features(torch.LongTensor(nodes).cuda())
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            G_ = G_.cuda()
            H_ = H_.cuda()
        else:
            # self_feats = self.features(torch.LongTensor(nodes))
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        X_ = self.hgnn(embed_matrix, G_)

        hedges_ = hedge_agg(H_, X_)
        unique_nodes_X_ = v_agg(H_, hedges_)
        nodes_embed = [unique_nodes_X_[unique_nodes[node], :] for node in nodes]

        nodes_embed = torch.stack(nodes_embed)
        nodes_embed = F.relu(nodes_embed.mm(self.weight))


        return nodes_embed

def construct_G(nodes, neighs, cuda=False):
    H = torch.zeros((len(nodes), len(neighs)))

    row_indices = [nodes[n] for neigh in neighs for n in neigh]
    column_indices = [i for i in range(len(neighs)) for _ in range(len(neighs[i]))]
    H[row_indices, column_indices] = 1
    # self-loop
    # diag = torch.ones(len(nodes))
    # diag = torch.diag(diag)
    #
    # H = torch.cat((H, diag), dim=0)
    G = generate_G_from_H(H)
    # G = torch.Tensor(G)
    return G, H

def generate_G_from_H(H, variable_weight=False, cuda=False):

    H = H.float()
    n_edge = H.shape[1]
    W = torch.ones(n_edge)
    # DV = torch.sum(H * W, axis=1)
    DV = torch.sum(H * W, dim=1)

    DE = torch.sum(H, dim=0, dtype=torch.float)
    invDE = torch.diag(torch.pow(DE, -1))
    DV2 = torch.diag(torch.pow(DV, -0.5))
    W = torch.diag(W)
    HT = H.T
    if cuda:
        DV2 = DV2.cuda()
        H = H.cuda()
        W = W.cuda()
        invDE = invDE.cuda()
        HT = HT.cuda()
        DV2 = DV2.cuda()
    if variable_weight:
        DV2_H = torch.mm(DV2, H)
        invDE_HT_DV2 = torch.mm(torch.mm(invDE, HT), DV2)
        return DV2_H, W, invDE_HT_DV2
    else:
        G = torch.mm(torch.mm(torch.mm(torch.mm(torch.mm(DV2, H), W), invDE), HT), DV2)
        return G

class HGNN_conv(nn.Module):
    """
    A HGNN layer
    """
    def __init__(self, dim_in, dim_out):
        super(HGNN_conv, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fc = nn.Linear(self.dim_in, self.dim_out, bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.LeakyReLU()


    def forward(self, feats, G):
        x = feats
        x = self.activation(self.fc(x))
        x = G.matmul(x)
        x = self.dropout(x)
        return x

def hedge_agg(H, X):
    DE = H.T.sum(1, keepdim=True)
    mask = H.T.div(DE)
    hedge = torch.mm(mask, X)
    return hedge

def v_agg(H, hedge):
    DV = H.sum(1, keepdim=True)
    mask = H.div(DV)
    vx = torch.mm(mask, hedge)
    return vx

class GraphSage(nn.Module):
	"""
	Vanilla GraphSAGE Model
	Code partially from https://github.com/williamleif/graphsage-simple/
	"""
	def __init__(self, num_classes, enc):
		super(GraphSage, self).__init__()
		self.enc = enc

	def forward(self, nodes):
		embeds = self.enc(nodes)
		return embeds


class MeanAggregator(nn.Module):
	"""
	Aggregates a node's embeddings using mean of neighbors' embeddings
	"""

	def __init__(self, features, cuda=False, gcn=False):
		"""
		Initializes the aggregator for a specific graph.

		features -- function mapping LongTensor of node ids to FloatTensor of feature values.
		cuda -- whether to use GPU
		gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
		"""

		super(MeanAggregator, self).__init__()

		self.features = features
		self.cuda = cuda
		self.gcn = gcn

	def forward(self, nodes, to_neighs, num_sample=10):
		"""
		nodes --- list of nodes in a batch
		to_neighs --- list of sets, each set is the set of neighbors for node in batch
		num_sample --- number of neighbors to sample. No sampling if None.
		"""
		# Local pointers to functions (speed hack)
		_set = set
		if not num_sample is None:
			_sample = random.sample
			samp_neighs = [_set(_sample(to_neigh,
										num_sample,
										)) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
		else:
			samp_neighs = to_neighs

		if self.gcn:
			samp_neighs = [samp_neigh.union(set([int(nodes[i])])) for i, samp_neigh in enumerate(samp_neighs)]
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		return to_feats


class Encoder(nn.Module):
    """
	Vanilla GraphSAGE Encoder Module
	Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, features, feature_dim,
				 embed_dim, adj_lists, aggregator,
				 num_sample=5,
				 base_model=None, gcn=False, cuda=False,
				 feature_transform=False):
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
			torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
		Generates embeddings for a batch of nodes.

		nodes     -- list of nodes
        """
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                        self.num_sample)

        if isinstance(nodes, list):
            index = torch.LongTensor(nodes)
        else:
            index = nodes

        if not self.gcn:
            if self.cuda:
                index = index.cuda()
                self_feats = self.features(index).cuda()
            else:
                self_feats = self.features(index)
            combined = torch.cat((self_feats, neigh_feats), dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.t()))
        return combined