# !/usr/bin/env python
# -*-coding:utf-8 -*-


import numpy as np
import torch.nn as nn
import torch
import scipy
import myhelper


def sigmoid(x, k):
    s = 1 - (k / (k + np.exp(x / k)))
    return s

class Generator(nn.Module):
    def __init__(self, args, device, user_num, item_num):
        super(Generator, self).__init__()
        self.args = args
        self.dataset = args.dataset
        self.latent_dim = args.hdim
        self.user_num = user_num
        self.item_num = item_num
        self.device = device
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.user_num, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.item_num, embedding_dim=self.latent_dim)
        torch.nn.init.normal_(self.embedding_user.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.item_total_emb_dim = self.latent_dim
        self.encoder = nn.Sequential(nn.Linear(self.item_total_emb_dim, 256, bias=True), nn.ReLU(), nn.Linear(256, self.item_total_emb_dim, bias=True), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(self.item_total_emb_dim, 256, bias=True), nn.ReLU(), nn.Linear(256, self.item_total_emb_dim, bias=True))

    def encode(self, item_id):
        batch_item_feature_embedded = self.embed_feature(item_id).float()
        batch_item_feature_encoded = self.encoder(batch_item_feature_embedded)
        return batch_item_feature_encoded

    def embed_feature(self, item_id):
        batch_item_feature_embedded = self.embedding_item.weight[item_id]
        return batch_item_feature_embedded

    def get_AllUserAndItemEmb(self):
        return self.embedding_user.weight, self.embedding_item.weight


class RFDAT(nn.Module):
    def __init__(self, args, n_user, n_item, interactionGraph, socialGraph, userdegree, itemdegree, device):
        super(RFDAT, self).__init__()
        self.args = args
        self.latent_dim = args.hdim
        self.inter_layers = args.ilayers
        self.social_layers = args.slayers
        self.num_users = n_user
        self.num_items = n_item
        self.interactionGraph = interactionGraph
        self.socialGraph = socialGraph
        self.device = device
        self.f = torch.nn.Sigmoid()
        self.dataset = args.dataset
        self.convergence = args.g_convergence
        self.link_topk = args.g_linktopk
        self.itemdegree = itemdegree
        sorted_item_degrees = sorted(itemdegree.items(), key=lambda x: x[0])
        _, item_degree_list = zip(*sorted_item_degrees)
        self.item_degree_numpy = np.array(item_degree_list)

        degree_to_item_list, degree_to_item_num = myhelper.degree_to_object(itemdegree, range(self.num_items))
        self.head_item, self.tail_item = myhelper.coldNodeProcess(degree_to_item_list, coldDegree=args.colddegree)
        self.tail_item_non01 = myhelper.coldNodeProcessNon01(degree_to_item_list, coldDegree=args.colddegree)
        degree_to_user_list, degree_to_user_num = myhelper.degree_to_object(userdegree, range(self.num_users))
        head_user, tail_user = myhelper.coldNodeProcess(degree_to_user_list, coldDegree=args.colddegree)
        self.head_user = torch.tensor(head_user).to(self.device)
        self.tail_user = torch.tensor(tail_user).to(self.device)
        self.generator = Generator(args, device, n_user, n_item)

    def compute_withItem_byAGE(self, item_r, item_c, enhance_weight):
        A = self.interactionGraph
        S = self.socialGraph
        users_emb, items_emb = self.generator.get_AllUserAndItemEmb()
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.inter_layers):
            all_emb_interaction = torch.sparse.mm(A, all_emb)
            users_emb_interaction, items_emb_interaction = torch.split(all_emb_interaction,
                                                                       [self.num_users, self.num_items])
            users_emb_next = users_emb_interaction
            items_emb_next = items_emb_interaction
            all_emb = torch.cat([users_emb_next, items_emb_next])
            embs.append(all_emb)
        final_embs = embs[-1]
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        users_embs_lists = [users]
        for layer in range(self.social_layers):
            users_emb_social = torch.sparse.mm(S, users)
            users_embs_lists.append(users_emb_social)
            users = users_emb_social
        users_embs_lists = torch.stack(users_embs_lists, dim=1)
        users_embs_final = torch.mean(users_embs_lists, dim=1)
        item_r_cpu = item_r.cpu()
        item_c_cpu = item_c.cpu()
        enhance_weight_cpu = enhance_weight.detach().cpu()
        item_graph = scipy.sparse.csr_matrix((enhance_weight_cpu, (item_r_cpu[0, :], item_c_cpu[0, :])),
                                             shape=(self.num_items, self.num_items))
        item_graph = myhelper._convert_sp_mat_to_sp_tensor(item_graph)
        item_graph = item_graph.coalesce().to(self.device)
        items_embs_lists = [items]
        item_embedding_enhanced = torch.sparse.mm(item_graph, items)
        items_embs_lists.append(item_embedding_enhanced)
        sum_weight = torch.from_numpy(
            self.convergence / (self.convergence + np.exp(self.item_degree_numpy / self.convergence)))
        sum_weight = sum_weight.to(self.device).float().unsqueeze(
            -1)
        items_embs_final = items_embs_lists[0] + sum_weight * items_embs_lists[1]
        return users_embs_final, items_embs_final


    def getembedding(self, fast_weights=None):
        if self.training:
            item_row_index, item_colomn_index, enhance_weight = self.item_link_predict_v1(self.tail_item, self.head_item, fast_weights)
        else:
            item_row_index, item_colomn_index, enhance_weight = self.item_link_predict_v1_forTest(self.tail_item, self.head_item)
        all_users, all_items = self.compute_withItem_byAGE(item_row_index, item_colomn_index, enhance_weight)
        self.final_user, self.final_item = all_users, all_items
        return all_users, all_items

    def forward(self, batch_user, batch_pos, batch_neg, fast_weights):
        all_users, all_items = self.getembedding(fast_weights=fast_weights)
        users_emb = all_users[batch_user.long()]
        pos_emb = all_items[batch_pos.long()]
        neg_emb = all_items[batch_neg.long()]
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb.unsqueeze(dim=1), neg_emb)
        neg_scores = torch.sum(neg_scores, dim=2)
        neg_scores = torch.mean(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        init_users_emb, init_items_emb = self.generator.get_AllUserAndItemEmb()
        userEmb0 = init_users_emb[batch_user.long()]
        posEmb0 = init_items_emb[batch_pos.long()]
        negEmb0 = init_items_emb[batch_neg.long()]
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(batch_user))
        return loss, reg_loss


    def getUsersRating(self, users):
        all_users, all_items = self.final_user, self.final_item
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def i2i(self, item1, item2, item_neg):
        mse_loss = nn.MSELoss()
        item1_embedded = self.generator.encode(item1)
        item2_embedded = self.generator.encode(item2)
        item_false_embedded = self.generator.encode(item_neg)
        i2i_score = torch.mm(item1_embedded, item2_embedded.permute(1, 0)).sigmoid()
        i2i_score_false = torch.mm(item1_embedded, item_false_embedded.permute(1, 0)).sigmoid()
        loss = (mse_loss(i2i_score, torch.ones_like(i2i_score)) + mse_loss(i2i_score_false, torch.zeros_like(i2i_score_false))) / 2
        return loss

    def item_link_predict_v1(self, tail_item, head_item, fast_weights):
        item_tail = torch.tensor(tail_item).to(self.device)
        item_top = torch.tensor(head_item).to(self.device)
        encoder_0_weight = fast_weights[0]
        encoder_0_bias = fast_weights[1]
        encoder_2_weight = fast_weights[2]
        encoder_2_bias = fast_weights[3]
        top_item_feature = self.generator.embed_feature(item_top).float()
        tail_item_feature = self.generator.embed_feature(item_tail).float()
        top_item_hidden = torch.mm(top_item_feature, encoder_0_weight.t()) + encoder_0_bias
        top_item_embedded = torch.mm(top_item_hidden, encoder_2_weight.t()) + encoder_2_bias
        tail_item_hidden = torch.mm(tail_item_feature, encoder_0_weight.t()) + encoder_0_bias
        tail_item_embedded = torch.mm(tail_item_hidden, encoder_2_weight.t()) + encoder_2_bias
        i2i_score = torch.mm(tail_item_embedded, top_item_embedded.permute(1, 0))
        i2i_score_masked, indices = i2i_score.topk(self.link_topk, dim=-1)
        tail_item_index = item_tail.unsqueeze(1).expand_as(i2i_score).gather(1, indices).reshape(-1)
        top_item_index = item_top.unsqueeze(0).expand_as(i2i_score).gather(1, indices).reshape(-1)
        row_index = tail_item_index.unsqueeze(0)
        colomn_index = top_item_index.unsqueeze(0)
        i2i_score_masked = i2i_score_masked.sigmoid()
        tail_item_degree = torch.sum(i2i_score_masked, dim=1)
        tail_item_degree = torch.pow(tail_item_degree + 1, -1).unsqueeze(1).expand_as(i2i_score_masked).reshape(-1)
        enhanced_value = i2i_score_masked.reshape(-1)
        joint_enhanced_value = enhanced_value * tail_item_degree
        enhance_weight = joint_enhanced_value
        return row_index, colomn_index, enhance_weight


    def item_link_predict_v1_forTest(self, tail_item, head_item):
        item_tail = torch.tensor(tail_item).to(self.device)
        item_top = torch.tensor(head_item).to(self.device)
        top_item_embedded = self.generator.encode(item_top)
        tail_item_embedded = self.generator.encode(item_tail)
        i2i_score = torch.mm(tail_item_embedded, top_item_embedded.permute(1, 0))
        i2i_score_masked, indices = i2i_score.topk(self.link_topk, dim=-1)
        tail_item_index = item_tail.unsqueeze(1).expand_as(i2i_score).gather(1, indices).reshape(-1)
        top_item_index = item_top.unsqueeze(0).expand_as(i2i_score).gather(1, indices).reshape(-1)
        row_index = tail_item_index.unsqueeze(0)
        colomn_index = top_item_index.unsqueeze(0)
        i2i_score_masked = i2i_score_masked.sigmoid()
        tail_item_degree = torch.sum(i2i_score_masked, dim=1)
        tail_item_degree = torch.pow(tail_item_degree + 1, -1).unsqueeze(1).expand_as(i2i_score_masked).reshape(-1)
        enhanced_value = i2i_score_masked.reshape(-1)
        joint_enhanced_value = enhanced_value * tail_item_degree
        enhance_weight = joint_enhanced_value
        return row_index, colomn_index, enhance_weight

