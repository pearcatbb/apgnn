# -*- coding: utf-8 -*-
# @Time     : 2023/1/25 19:27
# @Author   : Guo Jiayu
import pickle
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import loadmat
import copy as cp
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    confusion_matrix, classification_report, precision_score
from collections import defaultdict

"""
	Utility functions to handle data and evaluate model.
"""


def load_data(data, prefix='data/'):

    if data == 'amazon':
        data_file = loadmat(prefix + 'Amazon.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'amz_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'amz_upu_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_usu_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'amz_uvu_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    elif data == 'yelp':
        data_file = loadmat(prefix + 'YelpChi.mat')
        labels = data_file['label'].flatten()
        feat_data = data_file['features'].todense().A
        # load the preprocessed adj_lists
        with open(prefix + 'yelp_homo_adjlists.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rsr_adjlists.pickle', 'rb') as file:
            relation1 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rtr_adjlists.pickle', 'rb') as file:
            relation2 = pickle.load(file)
        file.close()
        with open(prefix + 'yelp_rur_adjlists.pickle', 'rb') as file:
            relation3 = pickle.load(file)
    elif data == 'elliptic':
        with open(prefix + 'elliptic.dat', 'rb') as file:
            data_file = pickle.load(file)

        # edge_index = data_file.edge_index
        # for i in range(len(edge_index[0])):
        #     if int(edge_index[1][i]) == 24318:
        #         print(int(edge_index[0][i]))
        labels = np.array(data_file.y, dtype='int')
        feat_data = data_file.x
        file.close()
        with open(prefix + 'elliptic.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        relation1 = None
        relation2 = None
        relation3 = None
    elif data == 'trans':
        data_file = pickle.load(open(prefix + 'weimao.pkl', 'rb'))
        labels = np.array(data_file['labels'], dtype='int')
        feat_data = data_file['features']
        homo = data_file['neighbor']
        relation1 = None
        relation2 = None
        relation3 = None
    elif data == 'tfinance':
        data_file = loadmat(prefix + 'tfinance.mat')
        labels = np.array(data_file['Label'].flatten(), dtype='int')
        feat_data = data_file['Attributes']
        # load the preprocessed adj_lists
        with open(prefix + 'tfinance.pickle', 'rb') as file:
            homo = pickle.load(file)
        file.close()
        relation1 = None
        relation2 = None
        relation3 = None


    return [homo, relation1, relation2, relation3], feat_data, labels


def normalize(mx):

    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_to_adjlist(sp_matrix, filename):

    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


def pos_neg_split(nodes, labels):

    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])

    return pos_nodes, neg_nodes


def node_balance(idx_train, y_train, adj_list, size):
    degree_train = [len(adj_list[node]) for node in idx_train]
    lf_train = (y_train.sum() - len(y_train)) * y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(idx_train, weights=smp_prob, k=size)


def test_apb(test_cases, labels, model, batch_size):

    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1 = 0.0
    acc = 0.00
    recall = 0.0
    label_list = []
    one_time = 0
    for iteration in range(test_batch_num):
        start_time = time.time()
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        if len(batch_nodes) == 0: break
        gnn_prob, label_prob1= model.to_prob(batch_nodes, batch_label, train_flag=False)
        end_time = time.time()
        one_time += end_time - start_time
        f1 += f1_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
        acc += accuracy_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1))
        recall += recall_score(batch_label, label_prob1.data.cpu().numpy().argmax(axis=1), average="macro")
        label_list.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())
    auc_label = roc_auc_score(labels, np.array(label_list))

    print("All Time =====================> " + str(one_time))
    print("Each Time=====================> " + str(one_time / len(labels)))
    print(f"F1-Macro: {f1 / test_batch_num:.4f}" +
          f"\tRecall: {recall / test_batch_num:.4f}\tAUC: {auc_label:.4f}\tAccuracy: {acc / test_batch_num:.4f}")
    # print(f"AUC: {auc_label:.4f}")
    # print(classification_report(labels, predict_all,digits=4))
    return auc_label, f1 / test_batch_num, recall / test_batch_num, acc / test_batch_num


def test_ap(test_cases, labels, model, batch_size):

    test_batch_num = int(len(test_cases) / batch_size) + 1
    f1 = 0.0
    acc = 0.00
    recall = 0.0
    label_list = []
    one_time = 0
    
    probs = torch.zeros((len(test_cases), 2))
    for iteration in range(test_batch_num):
        start_time = time.time()
        i_start = iteration * batch_size
        i_end = min((iteration + 1) * batch_size, len(test_cases))
        batch_nodes = test_cases[i_start:i_end]
        batch_label = labels[i_start:i_end]
        if len(batch_nodes) == 0: break
        # gnn_prob, label_prob = model.to_prob(batch_nodes, batch_label, train_flag=False)
        label_prob, gnn_prob = model.to_prob(batch_nodes, batch_label, train_flag=False)
        end_time = time.time()
        one_time += end_time - start_time

        prob = label_prob.softmax(1)
        probs[i_start: i_end] = prob.detach()
        # label_list.extend(label_prob1.data.cpu().numpy()[:, 1].tolist())
    f1, thres = get_best_f1(labels, probs)
    preds = torch.zeros(len(labels))
    preds[probs[:, 1] > thres] = 1
    # auc_label = roc_auc_score(labels, np.array(label_list))
    trec = recall_score(labels, preds, average='macro')
    tpre = precision_score(labels, preds, average='macro')
    tmf1 = f1_score(labels, preds, average='macro')
    tauc = roc_auc_score(labels, probs[:, 1].detach().numpy())
    acc = accuracy_score(labels, preds)
    # print("All Time =====================> " + str(one_time))
    # print("Each Time=====================> " + str(one_time / len(labels)))
    # print(f"F1-Macro: {tmf1:.4f}" +
    #       f"\tRecall: {trec:.4f}\tAUC: {tauc:.4f}\tAccuracy: {acc:.4f}")
    # print(f"AUC: {tauc:.4f}")
    # print(classification_report(labels, preds, digits=4))
    print(f"Precision:{tpre:.4f}\t Recall:{trec:.4f}\t F1:{tmf1:.4f}")
    return tauc, tmf1, trec, acc

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

