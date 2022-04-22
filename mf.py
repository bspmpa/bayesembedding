import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
from   torch.nn.init import normal_, xavier_normal_
import torch.nn.functional as F
import torch.nn as nn
from   tqdm import tqdm
import time
# import networkx as nx
import numpy as np
import scipy.sparse as sp
from   multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
import pdb
import pandas as pd

def get_matrix_row(mat):
    a = 1 / torch.sqrt(torch.sparse.sum(mat, dim = 0).to_dense())
    sqrt_norm_a = torch.sparse_coo_tensor([range(len(a)), range(len(a))], a)
    mat = torch.sparse.mm(mat, sqrt_norm_a)
    return mat

def get_matrix_line(mat): # get matrix in LINE for SVD
    vol = torch.sparse.sum(mat)
    a = 1 / torch.sparse.sum(mat, dim = 0).to_dense()
    b = vol / torch.sparse.sum(mat, dim = 1).to_dense()
    norm_a = torch.sparse_coo_tensor([range(len(a)), range(len(a))], a)
    norm_b = torch.sparse_coo_tensor([range(len(b)), range(len(b))], b)
    mat = torch.sparse.mm(mat, norm_a)
    mat = torch.sparse.mm(norm_b, mat)
    mat = torch.sparse_coo_tensor(mat._indices(), mat._values() - 1)
    mat = torch.log1p(mat) 
    return mat

def get_matrix_sym(mat): # get matrix for LightGCN
    a = 1 / torch.sqrt(torch.sparse.sum(mat, dim = 0).to_dense())
    b = 1 / torch.sqrt(torch.sparse.sum(mat, dim = 1).to_dense())
    sqrt_norm_a = torch.sparse_coo_tensor([range(len(a)), range(len(a))], a)
    sqrt_norm_b = torch.sparse_coo_tensor([range(len(b)), range(len(b))], b)
    mat = torch.sparse.mm(mat, sqrt_norm_a)
    mat = torch.sparse.mm(sqrt_norm_b, mat)
    return mat

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma
        self.beta  = nn.Parameter(torch.zeros(1))
    def forward(self, pos_scores, neg_scores, epoch=None):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_scores - neg_scores))
        return loss.mean()


class ListWiseLoss(nn.Module):
    def __init__(self):
        super(ListWiseLoss, self).__init__()
    def forward(self, pos_scores, neg_scores, epoch=None):
        scores = torch.cat([pos_scores, neg_scores], dim = 1)
        probs  = F.softmax(scores, dim=1)
        hit_probs = probs[:, 0]
        loss = -torch.log(hit_probs).mean()
        return loss


class PointWiseLoss(nn.Module):
    def __init__(self, alpha=None):
        super(PointWiseLoss, self).__init__()
        self.gamma = 1e-10
        self.alpha = alpha # nn.Parameter(torch.zeros(1))
    def forward(self, pos_score, neg_scores, epoch=None):
        pos_loss = -torch.log(self.gamma + torch.sigmoid(pos_score))
        if self.alpha is None:
            neg_loss = -torch.sum(torch.log(self.gamma + torch.sigmoid((-neg_scores))), dim = -1, keepdim = True)
            loss = (pos_loss + neg_loss) / (pos_score.shape[1] + neg_scores.shape[1])
        else:
            neg_loss = -torch.mean(torch.log(self.gamma + torch.sigmoid((-neg_scores))), dim = -1, keepdim = True)
            loss = self.alpha * pos_loss + (1 - self.alpha) * neg_loss
        return loss.mean()


def get_loss_func(names):
    names = names.lower().split("_")
    print("loss function:{}".format(names))
    if names[0] == "bpr":
        return BPRLoss()
    elif names[0] == "point":
        alpha = None
        if len(names) > 1:
            alpha = float(names[1]) 
        return PointWiseLoss(alpha)
    elif names[0] == "list":
        return ListWiseLoss()
    else: 
        raise ValueError("unknown loss: " + name)


def get_new_reg(item_emb):
    item_e = F.normalize(item_emb)
    mean_e = item_e.mean(0)
    out = sum(mean_e * mean_e)
    return out

   

class ItemCF(torch.nn.Module): # itemCF
    def __init__(self, config=None):
        super().__init__()
        self.name = "itemcf"
        self.mat  = config['mat']
        self.embedding_dim = config['embedding_dim']
        #a = 1 / torch.sqrt(torch.sparse.sum(self.mat, dim = 1).to_dense())
        #sqrt_norm = torch.sparse_coo_tensor([range(len(a)), range(len(a))], a)
        vec = get_matrix_row(self.mat)
        self.sim = torch.sparse.mm(vec.t(), vec)#.to_dense()  # item-item sim
        self.get_init_embeddings()
    
    def get_init_embeddings(self):
        print("svd help finish sim")
        start_time = time.time()
        u, s, vt = torch.svd_lowrank(self.sim, self.embedding_dim)
        u  = u * torch.sqrt(s)
        vt = vt * torch.sqrt(s)
        alpha = 0.75
        self.sim = (1 - alpha) * torch.mm(u, vt.t()) + alpha * self.sim
        print("done with time {:.2f} seconds".format(time.time() - start_time))
    
    def predict_one(self, test_sample, normed=True):
        user  = test_sample[:, 0].cpu()
        items = test_sample[:, 1:].cpu()
        score = torch.sparse.mm(self.mat, self.sim)  
        scores = torch.gather(score, 1, items)
        return scores
    
    def predict_all(self, normed=True):
        start_time = time.time()
        score = torch.sparse.mm(self.mat, self.sim)
        return score


class MF(torch.nn.Module): # MF
    def __init__(self, config):
        super().__init__()
        self.name = "mf"
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        self.num_neg = config['mat_num_neg']

    def forward(self, user, item, label=None):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        score = torch.mul(user_e, item_e).sum(-1)
        return score 
    
    def predict(self, user, item, label=None):
        return self.forward(user, item, label)
    
    def predict_all(self, normed=False):
        users = self.user_embedding.weight #.detach().cpu()
        items = self.item_embedding.weight #.detach().cpu()
        if normed:
            users = F.normalize(users)
            items = F.normalize(items)
        users = users.detach().cpu()
        items = items.detach().cpu()
        score = torch.matmul(users, items.t())
        return score
    
    def predict_one(self, test_sample, normed=False):
        users = self.user_embedding(test_sample[:, 0])              # (batch, 1,   dim)
        items = self.item_embedding(test_sample[:, 1:])  # (batch, 100, dim)
        if normed:
            users = F.normalize(users)
            items = F.normalize(items)
        score = torch.mul(users.unsqueeze(1), items).sum(-1)        # (batch, 100)
        return score
    
    def forward_all(self, user, pos_item, neg_items, normed=False):
        user_e      = self.user_embedding(user)
        pos_item_e  = self.item_embedding(pos_item)
        neg_items_e = self.item_embedding(neg_items)
        if normed:
            user_e      = F.normalize(user_e)
            pos_item_e  = F.normalize(pos_item_e)
            neg_items_e = F.normalize(neg_items_e)

        pos_score   = torch.mul(user_e, pos_item_e).sum(-1, keepdim = True)
        neg_scores  = torch.mul(user_e.unsqueeze(1), neg_items_e).sum(-1)
        return pos_score, neg_scores, 0 

class MFSVD(MF):  # MF with SVD init
    def __init__(self, config):
        super().__init__(config)
        self.name = "mfsvd"
        self.train_mat = config['mat']
        self.get_init_embeddings()
        
    def get_init_embeddings(self):
        #print("Get matrix for svd")
        mat = get_matrix_line(self.train_mat).to_dense() - 1 #np.log(5)
        #print("done. Start svd ...")
        u, s, vt = torch.svd_lowrank(mat, self.embedding_dim)
        user_emb = u  * torch.sqrt(s)
        item_emb = vt * torch.sqrt(s)
        #print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape)
        self.user_embedding = nn.Embedding.from_pretrained(user_emb, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(item_emb, freeze=False)
        #print("done.")


class MFProfile(MF):  # MF with SVD init
    def __init__(self, config):
        super().__init__(config)
        self.name = "MFProfile"
        self.train_mat = config['mat']
        self.get_init_embeddings()


    def get_init_embeddings(self):
        model = torch.load("/data/code_yang/cfbe/res/201/movielens/model.pt", map_location='cpu')
        # print(model)
        user_emb_ini = model['mf.user_embedding.weight']
        item_emb_ini = model['mf.item_embedding.weight']
        data = pd.read_csv("/data/code_yang/cfbe/be/movielens/data.csv")
        user_df = data.groupby('user').head(1)
        user_emb_ = torch.zeros((self.train_mat.shape[0], self.embedding_dim))
        item_emb_ = torch.zeros((self.train_mat.shape[1], self.embedding_dim))

        for (u,u_i) in list(zip(user_df['user'].values, user_df['user_prof_grine_ind'].values)):
            user_emb_[u,:] = user_emb_ini[u_i, :]
        for (i,i_i) in list(zip(user_df['item'].values, user_df['Genere_ind'].values)):
            item_emb_[i,:] = item_emb_ini[i_i, :]

        # print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape
        print(user_emb_[0,:])
        print(item_emb_[0,:])
        self.user_embedding = nn.Embedding.from_pretrained(user_emb_, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(item_emb_, freeze=False)
        # print("done.")

class SVD():  # svd
    def __init__(self, config):
        self.name = "svd"
        self.embedding_dim = config['embedding_dim']
        self.num_neg   = config['mat_num_neg']
        self.train_mat = config['mat']
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        self.dense = False
        self.self_train()
    
    def self_train(self, mat = "dw"):
        print("Get matrix for svd")
        mat = get_matrix_line(self.train_mat)
        if self.dense:
            mat = mat.to_dense() - np.log(self.num_neg)
        print("done. Start svd ...")
        print(mat)
        u, s, vt = torch.svd_lowrank(mat, self.embedding_dim)
        user_emb = u  * torch.sqrt(s)
        item_emb = vt * torch.sqrt(s)
        print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape)
        self.user_embedding = nn.Embedding.from_pretrained(user_emb, freeze=True)
        self.item_embedding = nn.Embedding.from_pretrained(item_emb, freeze=True)
        print("done.")
    
    def predict_all(self, normed=True):
        users = self.user_embedding.weight #.detach().cpu()
        items = self.item_embedding.weight #.detach().cpu()
        if normed:
            users = F.normalize(users)
            items = F.normalize(items)
        users = users.detach().cpu()
        items = items.detach().cpu()
        score = torch.matmul(users, items.t())
        return score
    
    def predict_one(self, test_sample, normed=True):
        user = test_sample[:, 0].cpu()
        item = test_sample[:, 1:].cpu()
        users = self.user_embedding(user)                           # (batch, 1,   dim)
        items = self.item_embedding(item)                           # (batch, 100, dim)
        if normed:
            users = F.normalize(users)
            items = F.normalize(items)
        score = torch.mul(users.unsqueeze(1), items).sum(-1)        # (batch, 100)
        return score

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)    

class NMF(nn.Module): # Neural CF
    def __init__(self, config):
        super().__init__()
        self.name = "nmf"
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.mf_embedding_size  = config['embedding_dim']
        self.mlp_embedding_size = config['embedding_dim']
        self.dropout_prob    = config['net_dropout']
        self.mlp_hidden_dims = config['hidden_units']
        self.user_embedding = nn.Embedding(self.num_user, self.mf_embedding_size)
        self.item_embedding = nn.Embedding(self.num_item, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.num_user, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.num_item, self.mlp_embedding_size)
        self.mlp = MLP(2 * self.mlp_embedding_size, self.mlp_hidden_dims, self.dropout_prob, False)
        self.out_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_dims[-1], 1)
        self.register_buffer("users", torch.arange(self.num_user))
     
    def get_score(self, user_e, item_e, user_mlp, item_mlp):
        mul_output = torch.mul(user_e, item_e) 
        mlp_output = self.mlp(torch.cat((user_mlp.expand_as(item_mlp), item_mlp), -1))
        score      = self.out_layer(torch.cat((mul_output, mlp_output), -1))
        return score.squeeze(-1)
    
    def predict_all(self, normed=False):
        scores = []
        item_w     = self.item_embedding.weight
        item_mlp_w = self.item_mlp_embedding.weight
        batch_size = 16
        for user in tqdm(self.users.split(batch_size)):
            user_e = self.user_embedding(user).unsqueeze(1)
            item_e = item_w.unsqueeze(0).expand(len(user), -1, -1)
            user_mlp_e = self.user_mlp_embedding(user).unsqueeze(1)
            item_mlp_e = item_mlp_w.unsqueeze(0).expand(len(user), -1, -1)
            #print("user=", user_e.shape, "item=", item_e.shape)
            score = self.get_score(user_e, item_e, user_mlp_e, item_mlp_e)
            scores.append(score.cpu())
        return torch.cat(scores, 0)
    
    def predict_one(self, test_sample, normed=False):
        scores = []
        batch_size = 256
        for test in tqdm(test_sample.split(batch_size)):
            user = test[:, 0]
            item = test[:, 1:]
            user_e = self.user_embedding(user).unsqueeze(1)
            item_e = self.item_embedding(item)
            user_mlp_e = self.user_mlp_embedding(user).unsqueeze(1)
            item_mlp_e = self.item_mlp_embedding(item)
            score  = self.get_score(user_e, item_e, user_mlp_e, item_mlp_e)
            scores.append(score.cpu())
        #print("one=", score.shape)
        return torch.cat(scores, 0)
        
    def forward_all(self, user, pos_item, neg_items, normed=False):
        user_e      = self.user_embedding(user).unsqueeze(1)
        pos_item_e  = self.item_embedding(pos_item).unsqueeze(1)
        neg_items_e = self.item_embedding(neg_items)
        user_mlp_e      = self.user_mlp_embedding(user).unsqueeze(1)
        pos_item_mlp_e  = self.item_mlp_embedding(pos_item).unsqueeze(1)
        neg_items_mlp_e = self.item_mlp_embedding(neg_items)
        pos_score   = self.get_score(user_e, pos_item_e, user_mlp_e, pos_item_mlp_e)
        neg_scores  = self.get_score(user_e, neg_items_e, user_mlp_e, neg_items_mlp_e)
        #print('pos_score', pos_score.shape, 'neg_scores', neg_scores.shape)
        return pos_score, neg_scores, 0

class MFBE(torch.nn.Module): # Bayesian embedding
    def __init__(self, config):
        super().__init__()
        self.name = "mfbe"
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.embedding_dim  = config['embedding_dim']
        self.user_mean = nn.Embedding(self.num_user, self.embedding_dim)
        self.user_std  = nn.Embedding(self.num_user, self.embedding_dim) 
        self.item_mean = nn.Embedding(self.num_item, self.embedding_dim)
        self.item_std  = nn.Embedding(self.num_item, self.embedding_dim)
        self.register_buffer("users", torch.arange(self.num_user))
        self.train_mat = config['mat']
        self.get_init_embeddings()
        self.num_sampling = 100
    
    def get_init_embeddings(self):
        nn.init.normal_(self.user_mean.weight, mean = 0.0, std = 0.001)
        nn.init.normal_(self.item_mean.weight, mean = 0.0, std = 0.001)
        nn.init.ones_(self.user_std.weight)
        nn.init.ones_(self.item_std.weight)
        print("done.")
        
    def get_user_embedding(self, user, num):  # (batch,) => (batch, num, dim) => (batch * num, 1, dim)
        #user_e = torch.zeros(len(user), num, self.embedding_dim).normal_()
        user_mean = self.user_mean(user).unsqueeze(1).expand(-1, num, -1)
        user_std  = self.user_std(user).unsqueeze(1).expand(-1,  num, -1) 
        noise  = torch.zeros_like(user_mean).normal_()
        #pdb.set_trace()
        user_e = user_mean + noise * user_std
        user_e = user_e.reshape(len(user) * num, 1, self.embedding_dim)
        return user_e
    
    def get_item_embedding(self, item, num):  # (batch,) => (batch, num,dim) => (batch * num, 1, dim)
        item_mean = self.item_mean(item).unsqueeze(1).expand(-1, num, -1)
        item_std  = self.item_std(item).unsqueeze(1).expand(-1, num, -1) 
        noise = torch.zeros_like(item_mean).normal_()
        item_e = item_mean + noise * item_std
        item_e = item_e.reshape(len(item) * num, 1, self.embedding_dim)
        return item_e
    
    def get_items_embedding(self, items, num):  # (batch, neg) => (batch, neg, dim) => (batch, num, neg, dim) => (batch * num, neg, dim)
        items_mean = self.item_mean(items).unsqueeze(1).expand(-1, num, -1, -1)
        items_std  = self.item_std(items).unsqueeze(1).expand(-1, num, -1, -1) 
        noise = torch.zeros_like(items_mean).normal_()
        items_e = items_mean + noise * items_std
        items_e = items_e.reshape(items.shape[0] * num, items.shape[1], self.embedding_dim)
        return items_e
    
    def get_score(self, user_e, item_e):
        return torch.sum(user_e * item_e, -1)
    
    def predict_all(self, normed=False):
        print("Running predict all ...")
        scores = []
        items_e_w = self.item_mean.weight      
        num_batch = 16
        for user in tqdm(self.users.split(num_batch)):
            user_e = self.user_mean(user).unsqueeze(1)
            item_e = items_e_w.unsqueeze(0).expand(len(user), -1, -1)
            #pdb.set_trace()
            score  = self.get_score(user_e, item_e)
            scores.append(score.cpu())
        return torch.cat(scores, 0)
    
    def predict_one(self, test_sample, normed=False):
        print("Running predict one ...")
        users = test_sample[:, 0]
        items = test_sample[:, 1:]
        scores = []
        items_e_w = self.item_mean.weight      
        num_batch = 256
        for user, item in tqdm(zip(users.split(num_batch), items.split(num_batch))):
            users_e = self.user_mean(user).unsqueeze(1)  # (batch, 1, dim)             
            items_e = self.item_mean(item)               # (batch, neg, dim)
            score = self.get_score(users_e, items_e)
            scores.append(score.cpu())
        return torch.cat(scores, 0)
    
    def forward_all(self, user, pos_item, neg_items, normed=False):
        user_e        = self.get_user_embedding(user, self.num_sampling) 
        pos_item_e    = self.get_item_embedding(pos_item, self.num_sampling)
        neg_items_e   = self.get_items_embedding(neg_items, self.num_sampling)
        #pdb.set_trace()
        pos_score   = self.get_score(user_e, pos_item_e)
        neg_scores  = self.get_score(user_e, neg_items_e)
        return pos_score, neg_scores, 0


class MFBESVD(MFBE): # Bayesian embedding through svd initialization
    def __init__(self, config):
        super().__init__(config)
        self.name = "mfbesvd"
        self.train_mat = config['mat']
        self.num_sampling = 100
        self.get_init_embeddings()
        
    
    def get_init_embeddings(self):
        print("Get matrix for svd")
        mat = get_matrix_line(self.train_mat).to_dense() - np.log(100)
        print("done. Start svd ...")
        u, s, vt = torch.svd_lowrank(mat, self.embedding_dim)
        user_emb = u  * torch.sqrt(s)
        item_emb = vt * torch.sqrt(s)
        #print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape)
        self.user_mean = nn.Embedding.from_pretrained(user_emb, freeze=False)
        self.item_mean = nn.Embedding.from_pretrained(item_emb, freeze=False)
        nn.init.ones_(self.user_std.weight)
        nn.init.ones_(self.item_std.weight)
        print("done.")


class MFBEProfile(MFBE):  # Bayesian embedding through svd initialization
    def __init__(self, config):
        super().__init__(config)
        self.name = "mfbeprofile"
        self.train_mat = config['mat']
        self.num_sampling = 100
        self.get_init_embeddings()

    def get_init_embeddings(self):
        model = torch.load("/data/code_yang/cfbe/res/201/movielens/model.pt", map_location='cpu')
        # print(model)
        user_emb_ini = model['mf.user_embedding.weight']
        item_emb_ini = model['mf.item_embedding.weight']
        data = pd.read_csv("/data/code_yang/cfbe/be/movielens/data.csv")
        user_df = data.groupby('user').head(1)
        user_emb_ = torch.zeros((self.train_mat.shape[0], self.embedding_dim))
        item_emb_ = torch.zeros((self.train_mat.shape[1], self.embedding_dim))

        for (u, u_i) in list(zip(user_df['user'].values, user_df['user_prof_grine_ind'].values)):
            user_emb_[u, :] = user_emb_ini[u_i, :]
        for (i, i_i) in list(zip(user_df['item'].values, user_df['Genere_ind'].values)):
            item_emb_[i, :] = item_emb_ini[i_i, :]

        # print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape
        # print(user_emb_[0, :])
        # print(item_emb_[0, :])
        self.user_embedding = nn.Embedding.from_pretrained(user_emb_, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(item_emb_, freeze=False)
        # print("done.")




class MFMBE(torch.nn.Module): # Mixture bayesian embedding
    def __init__(self, config):
        super().__init__()
        self.name = "mfmbe"
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']
        self.num_interest  = config['num_interest']
        self.user_alpha = nn.Embedding(self.num_user,  self.num_interest)
        self.user_mean  = nn.Embedding(self.num_user,  self.num_interest * self.embedding_dim)
        self.user_std   = nn.Embedding(self.num_user,  self.num_interest * self.embedding_dim) 
        self.item_mean  = nn.Embedding(self.num_item,  self.embedding_dim)
        self.item_std   = nn.Embedding(self.num_item,  self.embedding_dim)
        self.register_buffer("users", torch.arange(self.num_user))
        self.train_mat = config['mat']
        self.get_init_embeddings()
        self.num_sampling = 100
    
    def get_init_embeddings(self):
        nn.init.normal_(self.user_mean.weight, mean = 0.0, std = 0.001)
        nn.init.normal_(self.item_mean.weight, mean = 0.0, std = 0.001)
        nn.init.ones_(self.user_std.weight)
        nn.init.ones_(self.item_std.weight)
        nn.init.normal_(self.user_alpha.weight, mean = 0.0, std = 1.0)
        print("done.")
        
    def get_user_embedding(self, user, num):  # (batch,) => (batch, num, dim) => (batch * num, 1, dim)
        #user_e = torch.zeros(len(user), num, self.embedding_dim).normal_()
        user_a = torch.softmax(self.user_alpha(user), dim = 1).reshape(-1, 1, self.num_interest, 1).expand(-1, num, -1, self.embedding_dim) # (batch, num, k, dim)
        user_mean = self.user_mean(user).reshape(-1, self.num_interest, self.embedding_dim) # (batch, k*dim) => (batch, k, dim)
        user_mean = user_mean.unsqueeze(1).expand(-1, num, -1, -1)   # (batch, num, k, dim) 
        user_std  = self.user_std(user).reshape(-1, self.num_interest, self.embedding_dim)
        user_std  = user_std.unsqueeze(1).expand(-1,  num, -1, -1)   # (batch, num, k, dim)
        noise  = torch.zeros_like(user_mean).normal_()
        user_e = user_mean + noise * user_std   # (batch, num, k, dim)
        user_e = torch.sum(user_e * user_a, dim = 2)  # (batch, num, dim)
        user_e = user_e.reshape(len(user) * num, 1, self.embedding_dim)
        return user_e
    
    def get_item_embedding(self, item, num):  # (batch,) => (batch, num,dim) => (batch * num, 1, dim)
        item_mean = self.item_mean(item).unsqueeze(1).expand(-1, num, -1)
        item_std  = self.item_std(item).unsqueeze(1).expand(-1, num, -1) 
        noise  = torch.zeros_like(item_mean).normal_()
        item_e = item_mean + noise * item_std
        item_e = item_e.reshape(len(item) * num, 1, self.embedding_dim)
        return item_e
    
    def get_items_embedding(self, items, num):  # (batch, neg) => (batch, neg, dim) => (batch, num, neg, dim) => (batch * num, neg, dim)
        items_mean = self.item_mean(items).unsqueeze(1).expand(-1, num, -1, -1)
        items_std  = self.item_std(items).unsqueeze(1).expand(-1, num, -1, -1) 
        noise = torch.zeros_like(items_mean).normal_()
        items_e = items_mean + noise * items_std
        items_e = items_e.reshape(items.shape[0] * num, items.shape[1], self.embedding_dim)
        return items_e
    
    def get_score(self, user_e, item_e):
        return torch.sum(user_e * item_e, -1)
    
    def predict_all(self, normed=False):
        print("Running predict all ...")
        scores = []
        items_e_w = self.item_mean.weight      
        num_batch = 16
        for user in tqdm(self.users.split(num_batch)):
            user_e = self.user_mean(user).reshape(-1, self.num_interest, self.embedding_dim)  # (user, k * dim) => (user, k, dim)   #.unsqueeze(1)
            user_a = torch.softmax(self.user_alpha(user), dim = 1).unsqueeze(2)               # (user, k) => (item, k, 1)
            user_e = torch.sum(user_e * user_a, dim = 1, keepdims = True)                     # (user, 1, dim)
            item_e = items_e_w.unsqueeze(0).expand(len(user), -1, -1)                         # (user, item, dim) 
            #pdb.set_trace()
            score  = self.get_score(user_e, item_e)
            scores.append(score.cpu())
        return torch.cat(scores, 0)
    
    def predict_one(self, test_sample, normed=False):
        print("Running predict one ...")
        users = test_sample[:, 0]
        items = test_sample[:, 1:]
        scores = []
        items_e_w = self.item_mean.weight      
        num_batch = 256
        for user, item in tqdm(zip(users.split(num_batch), items.split(num_batch))):
            #pdb.set_trace()
            users_e = self.user_mean(user).reshape(-1, self.num_interest, self.embedding_dim)  # (batch, k, dim)
            users_a = torch.softmax(self.user_alpha(user), dim = 1).unsqueeze(2)               # (batch, k, 1)
            users_e = torch.sum(users_e * users_a, dim = 1, keepdims = True)                   # (batch, 1,   dim)
            items_e = self.item_mean(item)                                                     # (batch, neg, dim)
            score = self.get_score(users_e, items_e)
            scores.append(score.cpu())
        return torch.cat(scores, 0)
    
    def forward_all(self, user, pos_item, neg_items, normed=False):
        user_e        = self.get_user_embedding(user, self.num_sampling) 
        pos_item_e    = self.get_item_embedding(pos_item, self.num_sampling)
        neg_items_e   = self.get_items_embedding(neg_items, self.num_sampling)
        #pdb.set_trace()
        pos_score   = self.get_score(user_e, pos_item_e)
        neg_scores  = self.get_score(user_e, neg_items_e)
        return pos_score, neg_scores, 0    


class MFG(torch.nn.Module): # Gaussian embedding
    def __init__(self, config):
        super().__init__()
        self.name = "mfg"
        self.num_user = config['num_user']
        self.num_item = config['num_item']
        self.embedding_dim  = config['embedding_dim']
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.user_log_var   = nn.Embedding(self.num_user, self.embedding_dim) 
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        self.item_log_var   = nn.Embedding(self.num_item, self.embedding_dim)
        self.register_buffer("users", torch.arange(self.num_user))
        self.train_mat = config['mat']
        self.get_init_embeddings()
    
    def get_init_embeddings(self):
        print("Get matrix for svd")
        mat = get_matrix_line(self.train_mat).to_dense() - np.log(100)
        print("done. Start svd ...")
        u, s, vt = torch.svd_lowrank(mat, self.embedding_dim)
        user_emb = u  * torch.sqrt(s)
        item_emb = vt * torch.sqrt(s)
        #print("user_emb:", user_emb.shape, "item_emb:", item_emb.shape)
        self.user_embedding = nn.Embedding.from_pretrained(user_emb, freeze=False)
        self.item_embedding = nn.Embedding.from_pretrained(item_emb, freeze=False)
        nn.init.ones_(self.user_log_var.weight)
        nn.init.ones_(self.item_log_var.weight)
        print("done.")
    
    def get_mean_score(self, user_e, item_e):
        score = torch.mul(user_e, item_e)        
        return score.sum(-1)
    
    def get_score(self, user_e, item_e, user_var, item_var):
        #print("user_var=", user_var.shape, "item_var=", item_var.shape)
        user_var = user_var + 1e-10
        det   = torch.sum(torch.log(user_var), -1) - torch.sum(torch.log(item_var), -1)
        #print("det=", det.shape)
        trace = torch.sum(item_var / user_var, -1)
        #print("trace=", trace.shape)
        diff  = torch.sum((user_e - item_e) ** 2 / user_var, -1)
        #print("diff=", trace.shape)
        return 0.5 * (trace + diff - det - self.embedding_dim)
    
    
    def predict_all(self, normed=False):
        scores = []
        items_e_w       = self.item_embedding.weight      
        items_log_var_w = self.item_log_var.weight
        num_batch = 256
        for user in tqdm(self.users.split(num_batch)):
            user_e       = self.user_embedding(user).unsqueeze(1)
            user_log_var = self.user_log_var(user).unsqueeze(1)
            item_e       = items_e_w.unsqueeze(0).expand(len(user), -1, -1)
            item_log_var = items_log_var_w.unsqueeze(0).expand(len(user), -1, -1)
            score  = self.get_score(user_e, item_e, torch.exp(user_log_var), torch.exp(item_log_var))
            #score = self.get_mean_score(user_e, item_e)
            scores.append(score.cpu())
        return torch.cat(scores, 0)
    
    def predict_one(self, test_sample, normed=False):
        users = test_sample[:, 0]
        items = test_sample[:, 1:]
        users_e = self.user_embedding(users).unsqueeze(1)             
        items_e = self.item_embedding(items)        
        users_log_var = self.user_log_var(users).unsqueeze(1)             
        items_log_var = self.item_log_var(items)    
        score = self.get_score(users_e, items_e, torch.exp(users_log_var), torch.exp(items_log_var))
        #print("score=", score.shape)
        return score
    
    def forward_all(self, user, pos_item, neg_items, normed=False):
        user_e        = self.user_embedding(user).unsqueeze(1) 
        user_log_var  = self.user_log_var(user).unsqueeze(1)
        pos_item_e        = self.item_embedding(pos_item).unsqueeze(1)
        pos_item_log_var  = self.item_log_var(pos_item).unsqueeze(1) 
        neg_items_e       = self.item_embedding(neg_items)
        neg_items_log_var = self.item_log_var(neg_items) 
        pos_score   = self.get_score(user_e, pos_item_e,  torch.exp(user_log_var), torch.exp(pos_item_log_var))
        neg_scores  = self.get_score(user_e, neg_items_e, torch.exp(user_log_var), torch.exp(neg_items_log_var))
        return pos_score, neg_scores, 0


class LightGCN(nn.Module):  # LightGCN
    def __init__(self, config):
        super(LightGCN, self).__init__()
        self.num_user  = config['num_user']
        self.num_item  = config['num_item']
        self.num_layer = config['num_layer'] 
        self.embedding_dim  = config['embedding_dim']
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        self.register_buffer('A', get_matrix_sym(config['graph_matrix']))
        self.dropout = config["net_dropout"] 
        self.name = "lightngcn{}dropout{}".format(self.num_layer, self.dropout)
        
    def draw_graph(self):
        if self.dropout == 0:
            return self.A
        else:
            keep_prob = 1 - self.dropout
            size   = self.A.size()
            index  = self.A._indices()
            values = self.A._values()
            random_index = (torch.rand(len(values)) + keep_prob).int().bool()
            index  = index[:, random_index]
            values = values[random_index] / keep_prob
            return torch.sparse.FloatTensor(index, values, size)
    
    def get_embeddings(self, draw=True):
        if draw:
            g = self.draw_graph()
        else:
            g = self.A
        users = [self.user_embedding.weight]
        items = [self.item_embedding.weight]
        for layer in range(self.num_layer):
            users_layer = torch.sparse.mm(g, items[-1])
            items_layer = torch.sparse.mm(g.t(), users[-1])
            users.append(users_layer)
            items.append(items_layer)
        users_emb = torch.stack(users, dim = 1).mean(1)
        items_emb = torch.stack(items, dim = 1).mean(1)
        return users_emb, items_emb
        
    
    def get_reg_loss(self, user, user_emb, pos_item_emb, neg_item_emb):
        reg_loss = (1/2)*(user_emb.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) + neg_item_emb.norm(2).pow(2)) / float(len(user))
        return reg_loss
    
    def predict_all(self, normed=False):
        users_emb, items_emb = self.get_embeddings(False)
        if normed:
            users_emb = F.normalize(users_emb)
            items_emb = F.normalize(items_emb)
        users = users_emb.detach().cpu()
        items = items_emb.detach().cpu()
        score = torch.matmul(users, items.t())
        return score
    
    
    def predict_one(self, test_sample, normed=False):
        users_emb, items_emb = self.get_embeddings(False)
        users = users_emb[test_sample[:, 0]]  # (batch, 1,   dim)
        items = items_emb[test_sample[:, 1:]]             # (batch, 100, dim)
        if normed:
            users = F.normalize(users)
            items = F.normalize(items)
        score = torch.mul(users.unsqueeze(1), items).sum(-1)                      # (batch, 100)
        return score
    
    
    def forward_all(self, user, item, item_negs, normed=False):
        users_emb, items_emb = self.get_embeddings()
        user_e      = users_emb[user]
        pos_item_e  = items_emb[item]
        neg_items_e = items_emb[item_negs] 
        pos_score   = torch.mul(user_e, pos_item_e).sum(-1, keepdim = True)
        neg_scores  = torch.mul(user_e.unsqueeze(1), neg_items_e).sum(-1)
        reg_loss = self.get_reg_loss(user, user_e, pos_item_e, neg_items_e)
        return pos_score, neg_scores, reg_loss


class MFModel(nn.Module): 
    def __init__(self, model, loss="bpr", weight_decay = 0):
        super().__init__()
        self.mf = model
        self.loss = get_loss_func(loss)
        self.weight_decay = weight_decay
        print("weigh decay = {}".format(weight_decay))
        if not hasattr(model, "get_init_embeddings"):
            print("start init weight by MFModel")
            self.apply(self.init_weights)
        self.name =  "{}_{}_weightdecay_{}".format(model.name, loss, self.weight_decay)
    
    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean = 0.0, std = 0.001)
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias.data)
    
    def get_loss_params(self):
        if hasattr(self.loss, "beta"):
            return self.loss.beta.item()
        else:
            return None
    
    def forward(self, user, item):
        out = self.mf.forward_all(user, item)
        return out
    
    def predict(self, batch):
        user, items = batch
        user = user.reshape(-1, 1).expand_as(items)
        score = self.mf.predict(user, items)
        return user, score
    
    def get_cover_rate(self):
        if hasattr(self.mf, "item_embedding"):
            reg_loss = get_new_reg(self.mf.item_embedding.weight.detach())
            return 1 - reg_loss
        else:
            return None
    
    def predict_one(self, items, normed=False):
        return self.mf.predict_one(items, normed)
    
    def predict_all(self, normed=False):
        return self.mf.predict_all(normed)
        
    def get_loss(self, batch, normed=False, epoch=None):
        user, pos_item, items_neg = batch
        pos_scores, neg_scores, reg_loss = self.mf.forward_all(user, pos_item, items_neg, normed)
        loss = self.loss(pos_scores, neg_scores, epoch) + self.weight_decay * reg_loss
        return loss

    

    




