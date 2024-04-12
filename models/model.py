# -*- coding: utf-8 -*-
# @Time     : 2022/11/7 15:01
# @Author   : Guo Jiayu
from typing import List

import torch.nn as nn
import torch
from torch.nn import init
class APLayer(nn.Module):

	def __init__(self, num_classes, inter1, temperature):

		super(APLayer, self).__init__()
		self.inter1 = inter1
		self.xent = nn.CrossEntropyLoss()
		self.num_classes = num_classes
		self.weight = nn.Parameter(torch.FloatTensor(num_classes, inter1.embed_dim))
		init.xavier_uniform_(self.weight)

		self.classfier = nn.Linear(inter1.embed_dim * 2, num_classes)

		self.epsilon = 0.1
		self.prototype_shape = (7, inter1.embed_dim * 2)
		self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
											  requires_grad=True)
		self.num_prototypes = self.prototype_shape[0]
		self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
		self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
		self.k0 = 4

		self.temperature = temperature
		self.Softmax = nn.Softmax(dim=-1)

	def ini_prototype(self, prototype_shape, prototype_vectors, k0):
		self.prototype_shape = prototype_shape
		self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
											  requires_grad=True)
		self.prototype_vectors.data = prototype_vectors
		self.num_prototypes = self.prototype_shape[0]
		self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)
		self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)
		for j in range(self.num_prototypes):
			self.prototype_class_identity[j, 0 if j < k0 else 1] = 1
		self.k0 = k0

		self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

	def forward(self, nodes, labels, train_flag=True):
		embeds1, embeds2, embeds3 = self.inter1(nodes, labels, train_flag)
		# x = torch.cat((embeds1, embeds2, embeds3), dim=1)
		x = torch.cat((embeds1, embeds3), dim=1)
		prototype_activations, min_distances = self.prototype_distances(x)
		logits = self.last_layer(prototype_activations)
		scores = self.classfier(x)

		return scores, logits, x, min_distances

	def to_prob(self, nodes, labels, train_flag=True):
		scores, logits, x, min_distances = self.forward(nodes, labels, train_flag)
		gnn_scores = torch.sigmoid(scores)
		# label_scores = torch.sigmoid(label_logits)
		label_scores = torch.sigmoid(logits)
		return gnn_scores, label_scores

	def loss(self, nodes, labels, train_flag=True):
		embeds1, embeds2, embeds3 = self.inter1(nodes, labels, train_flag)
		# x = torch.cat((embeds1, embeds2, embeds3), dim=1)
		x = torch.cat((embeds1, embeds3), dim=1)
		scores = self.classfier(x)
		return scores, x

	def ConLoss(self):

		p_label = torch.zeros(self.num_prototypes)
		p_label[self.k0:] = 1
		p_label = p_label.contiguous().view(-1, 1)
		mask = torch.eq(p_label, p_label.T).float().cuda()

		anchor_dot_contrast = torch.div(torch.matmul(self.prototype_vectors, self.prototype_vectors.T), self.temperature)
		logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
		logits = anchor_dot_contrast - logits_max.detach()
		exp_logits = torch.exp(logits)

		logits_mask = torch.ones_like(mask) - torch.eye(self.num_prototypes).cuda()
		positives_mask = mask * logits_mask
		negatives_mask = 1. - mask

		num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数  [2 0 2 2]
		denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)
		# denominator = torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

		log_probs = logits - torch.log(denominator)

		log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
		loss = -log_probs
		# loss *= self.temperature
		loss = loss.mean()
		return loss

	def set_last_layer_incorrect_connection(self, incorrect_strength):
		'''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
		positive_one_weights_locations = torch.t(self.prototype_class_identity)
		negative_one_weights_locations = 1 - positive_one_weights_locations

		correct_class_connection = 1
		incorrect_class_connection = incorrect_strength
		self.last_layer.weight.data.copy_(
			correct_class_connection * positive_one_weights_locations
			+ incorrect_class_connection * negative_one_weights_locations)

	def prototype_distances(self, x):
		# x = x.cpu()
		xp = torch.mm(x, torch.t(self.prototype_vectors))
		distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
			torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
		similarity = torch.log((distance + 1) / (distance + self.epsilon))
		return similarity, distance


