# -*- coding: utf-8 -*-
# @Time     : 2023/1/31 14:05
# @Author   : Guo Jiayu
import time, datetime
import os
import random
import torch.nn.functional as F
import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable

from models.layers_a import InterAmazon
from models.layers_e import InterElliptic
from models.model import APLayer
from utils import load_data, pos_neg_split, normalize, node_balance, test_ap


class ModelHandler(object):

    def __init__(self, config):
        args = argparse.Namespace(**config)
        [homo, relation1, relation2, relation3], feat_data, labels = load_data(args.data_name, prefix=args.data_dir)

        # train_test split
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.data_name =='elliptic':
            #labels = labels[:10000]
            index = list(range(len(labels)))
            idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                                    train_size=args.train_ratio,
                                                                    random_state=2, shuffle=True)

            adj_lists = homo


        elif args.data_name =='amazon':
            index = list(range(3305, len(labels)))
            idx_train, idx_test, y_train, y_test = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                                    train_size=args.train_ratio, random_state=2,
                                                                    shuffle=True)
            adj_lists = [relation1, relation2, relation3]
        elif args.data_name == 'yelp':
            index = list(range(len(labels)))
            idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                                    train_size=args.train_ratio,
                                                                    random_state=2, shuffle=True)
            adj_lists = [relation1, relation2, relation3]
        elif args.data_name =='trans':
            index = list(range(len(labels)))
            idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                                    train_size=args.train_ratio,
                                                                    random_state=2, shuffle=True)
        elif args.data_name == 'tfinance':
            index = list(range(len(labels)))
            idx_train, idx_test, y_train, y_test = train_test_split(index, labels, stratify=labels,
                                                                    train_size=args.train_ratio,
                                                                    random_state=2, shuffle=True)
            adj_lists = homo
        print(f'Run on {args.data_name}, postive/total num: {np.sum(labels)}/{len(labels)}, train num {len(y_train)},' +
              f'test num {len(y_test)}, test positive num {np.sum(y_test)}')
        print(f"Feature dimension: {feat_data.shape[1]}")

        # split pos neg sets for under-sampling
        train_pos, train_neg = pos_neg_split(idx_train, y_train)

        feat_data = normalize(feat_data)

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id


        print(f'Model: {args.model}, emb_size: {args.emb_size}.')

        self.args = args
        self.dataset = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'homo': homo,
                        'idx_train': idx_train, 'idx_test': idx_test,
                        'y_train': y_train, 'y_test': y_test,
                        'train_pos': train_pos, 'train_neg': train_neg}


    def train(self):
        args = self.args
        feat_data, adj_lists = self.dataset['feat_data'], self.dataset['adj_lists']
        idx_train, y_train = self.dataset['idx_train'], self.dataset['y_train']
        idx_test, y_test =self.dataset['idx_test'], self.dataset['y_test']

        # initialize model input
        features = nn.Embedding(feat_data.shape[0], feat_data.shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
        if args.cuda:
            features.cuda()

        # build one-layer models
        if args.data_name == 'amazon' or args.data_name == 'yelp':
            inter = InterAmazon(features, feat_data.shape[1], args.emb_size,
                                adj_lists, homo=self.dataset['homo'], cuda=args.cuda)
        else:
            inter = InterElliptic(features, feat_data.shape[1], args.emb_size,
                                adj_lists, homo=self.dataset['homo'], cuda=args.cuda)

        gnn_model = APLayer(2, inter, args.temperature)

        if args.cuda:
            gnn_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
                                     weight_decay=args.weight_decay)

        timestamp = time.time()
        timestamp = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H-%M-%S')
        dir_saver = args.save_dir + timestamp
        path_saver = os.path.join(dir_saver, '{}_{}.pkl'.format(args.data_name, args.model))
        f1_mac_best, auc_best, ep_best = 0, 0, -1
        criterion = nn.CrossEntropyLoss()

        pre_train_epoch = args.pre_train_epoch
        pdist = nn.PairwiseDistance(p=2)
        for epoch in range(args.num_epochs):

            sampled_idx_train = node_balance(idx_train, y_train, self.dataset['homo'], size=len(self.dataset['train_pos']) * 2)
            list(set.union(set(sampled_idx_train), set(self.dataset['train_pos'])))
            random.shuffle(sampled_idx_train)

            num_batches = int(len(sampled_idx_train) / args.batch_size)

            loss = 0.0
            epoch_time = 0

            pretrain_x = torch.zeros((len(sampled_idx_train), args.emb_size*2))

            # mini-batch training
            for batch in range(num_batches):
                start_time = time.time()
                i_start = batch * args.batch_size
                i_end = min((batch + 1) * args.batch_size, len(sampled_idx_train))
                batch_nodes = sampled_idx_train[i_start:i_end]
                batch_label = self.dataset['labels'][np.array(batch_nodes)]

                optimizer.zero_grad()
                if epoch >= pre_train_epoch:
                    if args.cuda:
                        criterion = criterion.cuda()
                        scores, logits, x, min_distances = gnn_model(batch_nodes,
                                                                     Variable(torch.cuda.LongTensor(batch_label)))
                        prot_loss = criterion(logits.double(), torch.cuda.LongTensor(batch_label).squeeze())
                        gnn_loss = criterion(scores.double(), torch.cuda.LongTensor(batch_label).squeeze())
                    else:
                        scores, logits, x, min_distances = gnn_model(batch_nodes,
                                                                     Variable(torch.LongTensor(batch_label)))
                        prot_loss = criterion(logits, torch.LongTensor(batch_label).squeeze())
                        gnn_loss = criterion(scores, torch.LongTensor(batch_label).squeeze())
                    prototypes_of_correct_class = torch.t(gnn_model.prototype_class_identity[:, batch_label].bool())
                    split = torch.unique(torch.where(prototypes_of_correct_class == True)[0], return_counts=True)[1]
                    simi_index = [int(torch.min(scores, 0)[1]) for scores in torch.split(min_distances[prototypes_of_correct_class], split.tolist())]
                    simi_index = batch_label + batch_label * (gnn_model.k0 - 1) + simi_index

                    true_prot = gnn_model.prototype_vectors[simi_index]
                    # tt = pdist(x,true_prot)
                    # cluster_cost = torch.mean(torch.sum(torch.mul(x, true_prot), 1))
                    cluster_cost = torch.mean(pdist(x,true_prot))
                    ld = 0
                    for k in range(2):
                        if k == 0:
                            p = gnn_model.prototype_vectors[: gnn_model.k0]
                        else:
                            p = gnn_model.prototype_vectors[gnn_model.k0: ]
                        p = F.normalize(p, p=2, dim=1)
                        p = p.cpu()
                        matrix1 = torch.mm(p, torch.t(p)) - torch.eye(p.shape[0]) - 0.1
                        matrix2 = torch.zeros(matrix1.shape)
                        ld += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))

                    # conloss = gnn_model.ConLoss()

                    # loss = 0.5 * gnn_loss + 1.5 * prot_loss + args.lamda1 * conloss + args.lamda2 * cluster_cost + args.gama * ld
                    loss = 0.5 * gnn_loss + 1.5 * prot_loss  + args.lamda2 * cluster_cost + args.gama * ld
                    loss.backward()
                    optimizer.step()
                    end_time = time.time()
                    epoch_time += end_time - start_time
                    loss += loss.item()
                else:
                    if args.cuda:
                        criterion = criterion.cuda()
                        scores, x = gnn_model.loss(batch_nodes, Variable(torch.cuda.LongTensor(batch_label)))
                        gnn_loss = criterion(scores.double(), torch.cuda.LongTensor(batch_label).squeeze())
                    else:
                        scores, x = gnn_model.loss(batch_nodes, Variable(torch.LongTensor(batch_label)))
                        gnn_loss = criterion(scores, torch.LongTensor(batch_label).squeeze())
                    loss = gnn_loss
                    loss.backward()
                    optimizer.step()
                    end_time = time.time()
                    epoch_time += end_time - start_time
                    loss += loss.item()
                    if epoch == pre_train_epoch - 1:
                        pretrain_x[i_start: i_end] = x
            if epoch == pre_train_epoch - 1:


                train_labels = self.dataset['labels'][np.array(sampled_idx_train)]

                y0 = np.where(np.array(train_labels) == 0)[0]
                y1 = np.where(np.array(train_labels) == 1)[0]

                embeddings_0 = pretrain_x[y0, :].detach().numpy()
                embeddings_1 = pretrain_x[y1, :].detach().numpy()

                k0, means0 = self.get_gaussian_prot(embeddings_0)
                k1, means1 = self.get_gaussian_prot(embeddings_1)
                prototype_shape = (k0 + k1, args.emb_size * 2)
                weight = np.zeros(prototype_shape)
                weight[0: k0] = means0
                weight[k0:] = means1
                gnn_model.ini_prototype(prototype_shape, torch.tensor(weight).float(), k0)
                if args.cuda:
                    gnn_model.cuda()
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gnn_model.parameters()), lr=args.lr,
                                             weight_decay=args.weight_decay)

                # saver = os.path.join('./middle_models/', '{}_{}.pkl'.format(args.data_name, args.model))
                # torch.save(gnn_model.state_dict(), saver)
            # print(f'Epoch: {epoch // 5}, loss: {loss:.8f}, time: {epoch_time:.3f}s')

            # Valid the model for every $valid_epoch$ epoch
            if epoch % args.valid_epochs == 0 and epoch >= pre_train_epoch:
                print("Valid at epoch {}".format(epoch - 20))
                # print(f'Epoch: {epoch // 5}, loss: {losses:.8f}, time: {epoch_time:.3f}s')
                auc_label, f1, recall, acc = test_ap(idx_test, y_test, gnn_model,
                                                                                args.batch_size)
                if auc_label > auc_best:
                    f1_mac_best, auc_best, ep_best = f1, auc_label, epoch
                    if not os.path.exists(dir_saver):
                        os.makedirs(dir_saver)
                    print('  Saving model ...')
                    torch.save(gnn_model.state_dict(), path_saver)
        # path_saver = "./pytorch_models/2023-02-09 10-43-36/blogcatelog_DualGNN.pkl"
        # path_saver = "./pytorch_models/2023-03-30 13-34-57/trans_APGNN.pkl"
        print("Restore model from epoch {}".format(ep_best - 20))
        print("Model path: {}".format(path_saver))
        gnn_model.load_state_dict(torch.load(path_saver))
        auc_label, f1, recall, acc = test_ap(idx_test, y_test, gnn_model, args.batch_size)
        return auc_label, f1, recall, acc
    def get_gaussian_prot(self, x):

        max_k = 0
        max_score = -1
        time = 0
        for i in range(2, 10):

            n_clusters = i
            cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
            score = silhouette_score(x, cluster.labels_)
            if score > max_score:
                max_k = i
                max_score = score
                means = cluster.cluster_centers_
                time = 0
            else:
                time += 1
            if time > 2:
                break
        return max_k, means
        # min_k = 0
        # min_aic = float('inf')
        # # early stop
        # time = 0
        # means = None
        # for i in range(2, 10):
        #
        #     n_clusters = i
        #     cluster = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0).fit(x)
        #     aic = cluster.aic(x)
        #     if aic < min_aic:
        #         min_k = i
        #         min_aic = aic
        #         means = cluster.means_
        #         time = 0
        #     else:
        #         time += 1
        #     if time > 2:
        #         break
        # return min_k, means