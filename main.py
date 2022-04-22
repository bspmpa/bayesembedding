import os
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import logging
import yaml
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse as sp
from mf import MFModel, MF, NMF, LightGCN, ItemCF, MFSVD, SVD, MFG, MFBE, MFBESVD, MFMBE, MFProfile, MFBEProfile
import random
from tqdm import tqdm
from numba import jit
import pdb
import time
import psutil
from collections import defaultdict
import pynvml
import numba

pynvml.nvmlInit()
max_gradient_norm = 10.0
# pip install nvidia-ml-py

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="mfbeprofile", help="mf, light, ..., mfs")
parser.add_argument("--device", "-cpu", type=int, default=1)
parser.add_argument("--dataset", "-n", type=str, default="movielens", help="pinterest or movielens or gowalla or yelp2018 or amazon_book")
parser.add_argument("--out_dir", "-o", type=int, default=2012)
parser.add_argument("--epoch", "-e", type=int, default=100)
parser.add_argument("--batch_size", "-b", type=int, default=1024)
parser.add_argument("--learning_rate", "-lr", type=float, default=0.001)
parser.add_argument("--weight_decay", "-w", type=float, default=0)
parser.add_argument("--embedding_dim", "-dim", type=int, default=64)
parser.add_argument("--num_layer", "-layer", type=int, default=3)
parser.add_argument("--train_type", "-t", type=str, default="active", help="active or fixed")
parser.add_argument("--num_negative", "-g", type=int, default=50)
parser.add_argument("--test_type", "-test", type=str, default="all", help="all or one")
parser.add_argument("--patience", "-p", type=int, default=10, help="the patience in early stop")
parser.add_argument("--loss_name", "-loss", type=str, default="bpr", help="bpr, point, list, cosine, hinge")
parser.add_argument("--seq_g", "-sg", type=float, default=1.0)
parser.add_argument("--net_dropout", "-net_drop", type=float, default=0)
parser.add_argument("--emb_dropout", "-emb_drop", type=float, default=0)
parser.add_argument("--noise", "-noise", type=float, default=0.1)
parser.add_argument("--seed", "-seed", type=int, default=0)
parser.add_argument("--normed", "-norm", action="store_true")
parser.add_argument("--svd", "-svd", action="store_true")
parser.add_argument("--bias", "-bias", action="store_true")
parser.add_argument("--use_dnn", "-use_dnn", action="store_true")
parser.add_argument("--optimizer", "-opt", type=str, default="adam", help="adam, sgd")
parser.add_argument("--sample_alpha", "-alpha", type=float, default=0)
parser.add_argument("--topk", "-k", type=int, default=10)
parser.add_argument("--mc_sample", type=int, default=30)
args = parser.parse_args()


def get_uncertainty_new(model, num_user, num_item, embedding_dim, mc_sample, device, sample_num=1000, ana=True):
    if sample_num < num_user:
        user_sel = random.sample(range(num_user), sample_num)
    else:
        user_sel = range(num_user)

    # if sample_num<num_item:
    #     item_sel = random.sample(range(num_item), sample_num)
    # else:
    item_sel = range(num_item)

    user = torch.arange(num_user).to(device)[user_sel]
    item = torch.arange(num_item).to(device)

    item_mean = model.mf.item_mean(item).unsqueeze(0).expand(mc_sample, -1, -1)
    item_std = model.mf.item_std(item).unsqueeze(0).expand(mc_sample, -1, -1)
    noise = torch.zeros_like(item_mean).normal_()
    item_e = item_mean + noise * item_std

    user_mean = model.mf.user_mean(user).unsqueeze(0).expand(mc_sample, -1, -1)
    user_std = model.mf.user_std(user).unsqueeze(0).expand(mc_sample, -1, -1)
    noise = torch.zeros_like(user_mean).normal_()
    user_e = user_mean + noise * user_std

    rand_mat = torch.matmul(user_e, item_e.permute(0, 2, 1)).detach().cpu().numpy()

    rand_mat_ = rand_mat.reshape((mc_sample, -1))

    logits_arr = 1 / (1 + np.exp(-rand_mat_))

    entropy = -1 * np.mean(logits_arr, axis=0) * np.log(1e-8 + np.mean(logits_arr, axis=0))
    aleatoric = -1 * np.mean(logits_arr * np.log(logits_arr + 1e-8), axis=0)
    epistemic = entropy - aleatoric

    if ana == True:
        for name, param in model.mf.named_parameters():
            if name == 'user_std.weight':
                user_std = param.data.detach().cpu().numpy()
            if name == 'item_std.weight':
                item_std = param.data.detach().cpu().numpy()
            if name == 'user_mean.weight':
                user_mean = param.data.detach().cpu().numpy()
            if name == 'item_mean.weight':
                item_mean = param.data.detach().cpu().numpy()
        sig = []
        iii = []
        for i in range(len(user_sel)):
            for j in range(item_mean.shape[0]):
                iii.append((i, j))
                sig_ = (user_mean[i, :] ** 2 * user_std[i, :]).sum() + (item_mean[j, :] ** 2 * item_std[j, :]).sum() + (
                            user_std[i, :] * item_std[j, :]).sum()
                sig.append(sig_)

        return entropy.mean(), aleatoric.mean(), epistemic.mean(), rand_mat, user_sel, item_sel, sig, iii
    else:
        return entropy.mean(), aleatoric.mean(), epistemic.mean(), rand_mat, user_sel, item_sel


def predictive_uncertainty(model, mc_sample, device, test_sample):
    user = test_sample[:, 0].to(device)
    item = test_sample[:, 1].to(device)

    item_mean = model.mf.item_mean(item).unsqueeze(0).expand(mc_sample, -1, -1)
    item_std = model.mf.item_std(item).unsqueeze(0).expand(mc_sample, -1, -1)
    noise = torch.zeros_like(item_mean).normal_()
    item_e = item_mean + noise * item_std

    user_mean = model.mf.user_mean(user).unsqueeze(0).expand(mc_sample, -1, -1)
    user_std = model.mf.user_std(user).unsqueeze(0).expand(mc_sample, -1, -1)
    noise = torch.zeros_like(user_mean).normal_()
    user_e = user_mean + noise * user_std

    rand_mat = torch.sum(user_e * item_e, 2, keepdim=False).detach().cpu().numpy()
    logits_arr = 1 / (1 + np.exp(-rand_mat))

    entropy = -1 * np.mean(logits_arr, axis=0) * np.log(1e-8 + np.mean(logits_arr, axis=0))
    aleatoric = -1 * np.mean(logits_arr * np.log(logits_arr + 1e-8), axis=0)
    epistemic = entropy - aleatoric

    return entropy, aleatoric, epistemic, rand_mat


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_weights(model, checkpoint):
    torch.save(model.state_dict(), checkpoint)


def load_weights(checkpoint, device):
    return torch.load(checkpoint, map_location=device)


def count_parameters(model, count_embedding=True):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("{} \t params:{:d}".format(name, param.numel()))
            total_params += param.numel()
    print("Total number of parameters: {}".format(total_params))


def get_memory_info():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024 ** 3


def get_gpu_info():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_size = meminfo.total / 1024 ** 3
    use_size = meminfo.used / 1024 ** 3
    free_size = meminfo.free / 1024 ** 3
    gpu_name = str(pynvml.nvmlDeviceGetName(handle), "utf-8")
    print(
        "GPU({:s}): total = {:.2f}G, used = {:.2f}G, free = {:.2f}G".format(gpu_name, total_size, use_size, free_size))


def get_mean_ndcg(label, total, topk=20):
    s = torch.log2(torch.arange(2, topk + 2))
    dcg_vals = torch.sum(label / s, dim=1)
    w = torch.tensor([1e-10, *torch.cumsum(1 / s, 0).tolist()])
    idcg_vals = w[torch.clamp(total, 0, topk).long()]
    return torch.mean(dcg_vals / idcg_vals)


class LogLoss(nn.Module):
    def forward(self, y_pred, y_true):
        eps = 1e-7
        r = -(torch.log(y_pred + eps) * y_true + torch.log(1 - y_pred + eps) * (1 - y_true)).mean()
        return r


def get_logger(log_path="/code/r9/be/matching.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s:%(message)s'))
    logger.addHandler(stream_handler)
    return logger


class EarlyStopper():
    def __init__(self, num_trials=20):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_result = 0
        self.best_model = []

    def is_continuable(self, result, model):
        if result > self.best_result:
            self.best_result = result
            self.trial_counter = 0
            self.best_model = model
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


class Evaluation():
    def __init__(self, mat_all, mat_train, test_user_total, topk=10):
        self.mat = mat_all.to_dense()
        self.test_user_total = torch.tensor(test_user_total)
        self.train_mat = mat_train * 10000
        self.topk = topk

    def get_one(self, score):
        _, w = torch.where(torch.argsort(score, descending=True) == 0)
        log = -torch.mean(torch.log(torch.sigmoid(score[:, 0])))
        hit = torch.mean((w < self.topk).float())
        ndcg = torch.mean((1 / torch.log2(w + 2)) * (w < self.topk))
        mr = w.float().mean()
        return log.item(), mr.item(), hit.item(), ndcg.item()

    def get_all(self, score):
        score_ = torch.sigmoid(score) - self.train_mat
        # pdb.set_trace()
        test_score, s = torch.topk(score_, self.topk)
        test_label = torch.gather(self.mat, 1, s) / 2
        if len(torch.where(test_label == 0.5)[0]) != 0:
            print(score)
        assert len(torch.where(test_label == 0.5)[0]) == 0
        ndcg = get_mean_ndcg(test_label, self.test_user_total)
        hr = torch.mean((test_label.sum(-1) > 0).float())
        logloss = -torch.mean(torch.log(test_score[torch.where(test_label == 1.0)] + 1e-10))
        recall = torch.mean(test_label.sum(-1) / self.test_user_total)
        return logloss.item(), recall.item(), hr.item(), ndcg.item()


def get_train_instances(mat, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users, num_items = mat.shape
    for (u, i) in tqdm(mat.keys()):
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in mat:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels


def get_prob(users, items, alpha=1):
    df = pd.DataFrame(data={'user': users, 'item': items})
    num = df.groupby("item").size().values
    out = num ** alpha
    return out / out.sum()


class Dataset_CF(object):
    def __init__(self, users, items, num_neg, alpha=0):
        self.users = users
        self.items = items
        self.num_negative = num_neg
        self.num_item = len(np.unique(items))
        self.mat = sp.coo_matrix((np.ones_like(users), (users, items)), dtype=np.float32).todok()
        self.prob = get_prob(users, items, alpha)
        self.sampling = self.get_negs_ if alpha == 0 else get_negs
        # print(self.prob)

    def get_negs_(self, user):
        items_neg = np.zeros(self.num_negative, dtype=np.int64)
        s = 0
        while s < self.num_negative:
            js = np.random.randint(0, self.num_item, (int(1.2 * self.num_negative),))
            for j in js:
                if ((user, j) not in self.mat) and (s < self.num_negative):
                    items_neg[s] = j
                    s = s + 1
        return items_neg

    def get_negs_old(self, user):
        items_neg = []
        for t in range(self.num_negative):
            j = np.random.randint(self.num_item)
            while (user, j) in self.mat:
                j = np.random.randint(self.num_item)
            items_neg.append(j)
        return np.array(items_neg)

    def get_negs(self, user):
        items_neg = np.zeros(self.num_negative, dtype=np.int64)
        s = 0
        while s < self.num_negative:
            js = np.random.choice(self.num_item, int(1.2 * self.num_negative), p=self.prob, replace=True)
            # js = np.random.randint(0, self.num_item, int(1.2 * self.num_negative))
            for j in js:
                if ((user, j) not in self.mat) and (s < self.num_negative):
                    items_neg[s] = j
                    s = s + 1
        return items_neg

    def __getitem__(self, index):
        user = self.users[index]
        pos_item = self.items[index]
        neg_items = self.get_negs(user)
        # print(user, neg_items)
        return user, pos_item, neg_items

    def __len__(self):
        return len(self.users)


def get_test_sample_items(user_pos_item, mat, num_negatives):
    out = []
    num_users, num_items = mat.shape
    for (u, i) in user_pos_item.items():
        user_input = []
        user_input.append(u)
        user_input.append(i)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in mat:
                j = np.random.randint(num_items)
            user_input.append(j)
        out.append(user_input)
    return torch.from_numpy(np.array(out))


def get_svd_mat(users, items, num_user, num_item, num_negative):
    # mat  = sp.coo_matrix((np.ones_like(users), (users, items)), dtype=np.float32).todok()
    start_time = time.time()
    print("begin svd matrix")
    out = sp.coo_matrix((np.ones(len(users)), (users, items)), dtype=np.float32).todense()
    s = 0
    num_total = len(users)
    for i in tqdm(range(num_negative)):
        u_list = np.random.randint(0, num_user, num_total)
        i_list = np.random.randint(0, num_item, num_total)
        out[u_list, i_list] = out[u_list, i_list] * 2 - 1
    out[out < 0] = -1
    pos_v, neg_v = np.sum(out == 1), np.sum(out == -1)
    print("pos:{}, neg:{}, neg/pos:{:.2f}".format(pos_v, neg_v, neg_v / pos_v))
    out = sp.coo_matrix(out)
    mat = torch.sparse_coo_tensor((out.row, out.col), out.data, (num_user, num_item), dtype=torch.float32)
    print("done with {:.2f} seconds".format(time.time() - start_time))
    return mat


def get_mf_train(name="pinterest", batch_size=1024, num_negative=10, svd=False, sample_alpha=0,
                 user_index='user', item_index='item'):
    data = pd.read_csv("/data/code_yang/cfbe/be/{}/data.csv".format(name))
    train_data = data[data['type'] == 1]
    test_data = data[data['type'] == 2]
    num_user = data[user_index].max() + 1
    num_item = data[item_index].max() + 1
    train_dataset = Dataset_CF(train_data[user_index].values, train_data[item_index].values, num_negative, sample_alpha)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)

    train_mat = torch.sparse_coo_tensor(train_data[[user_index, item_index]].values.T, train_data['type'].values,
                                        (num_user, num_item), dtype=torch.float32)
    # pdb.set_trace()
    mat = torch.sparse_coo_tensor(data[[user_index, item_index]].values.T, data['type'].values, (num_user, num_item),dtype=torch.float32)
    dok_mat = sp.coo_matrix((np.ones(len(data[user_index])), (data[user_index].values, data[item_index].values)),dtype=np.float32).todok()
    print("dataset:", name)
    print("num of user = {} \t num of item = {}".format(num_user, num_item))
    print("train sample:{}  \t test sample:{}".format(len(train_dataset), len(data) - len(train_dataset)))
    print("data dense: {:.2f}%".format(100 * len(data) / (num_item * num_user)))
    user_total = np.zeros(num_user) + 1e-10
    test_user_len = test_data.groupby(user_index).size()
    user_total[test_user_len.index] = test_user_len.values
    test_pos_item = test_data.groupby(user_index).min("time")[item_index]
    test_sample_items = get_test_sample_items(test_pos_item, dok_mat, 100)
    svd_mat = None
    print("test, totally {} users, mean interactions: {:.2f}".format(len(test_user_len), user_total.mean()))
    return num_user, num_item, train_loader, mat, train_mat, user_total, test_sample_items, svd_mat


def get_train_new(name, batch_size, num_neg=10, train_type="active", matrix_type="sym", svd=False, sample_alpha=0):
    num_user, num_item, train_loader, mat, train_mat, user_total, test_sample, svd_mat = get_mf_train(name, batch_size,
                                                                                                      num_neg)
    return num_user, num_item, train_loader, mat, train_mat, user_total, test_sample, svd_mat


''' -------------------------- MODEL -------------------------'''


def get_model(name, config, loss, train_type="active", weight_decay=0):
    name = name.lower()
    if name == 'mf':
        model = MF(config)
    elif name == 'nmf':
        print(config)
        model = NMF(config)
    elif name == 'light':
        model = LightGCN(config)
    elif name == "itemcf":
        model = ItemCF(config)
    elif name == "mfsvd":
        model = MFSVD(config)
    elif name == "svd":
        model = SVD(config)
    elif name == "mfg":
        model = MFG(config)
    elif name == "mfbe":
        model = MFBE(config)
    elif name == "mfbesvd":
        model = MFBESVD(config)
    elif name == "mfmbe":
        model = MFMBE(config)
    elif name == "mfprofile":
        model = MFProfile(config)
    elif name == "mfbeprofile":
        model = MFBEProfile(config)
    else:
        raise ValueError('unknown model name: ' + name)
    return MFModel(model, loss, weight_decay)


def to_device(data, device):
    out = [v.to(device) for v in data]
    return out


def test_one(model, items, evaluate, normed):
    model.eval()
    targets = []
    with torch.no_grad():
        items = items.to(device)
        score = model.predict_one(items, normed)
        targets.append(score.detach().cpu())
    scores = torch.cat(targets, dim=0)
    log, mr, hit, ndcg = evaluate.get_one(scores)
    return {"logloss_one": log, "recall_one": hit, "hr_one": hit, "ndcg_one": ndcg}


def test_all(model, evaluate, normed):
    model.eval()
    targets = []
    res = {"logloss_all": -1, "recall_all": -1, "hr_all": -1, "ndcg_all": -1}
    with torch.no_grad():
        score = model.predict_all(normed)
        if score is not None:
            log, recall, hr, ndcg = evaluate.get_all(score.cpu())
            res = {"logloss_all": log, "recall_all": recall, "hr_all": hr, "ndcg_all": ndcg}
    return res


def test(model, items, evaluate, test_type, normed):
    print("eval normalized:{}".format(normed))
    if test_type == "one":
        return test_one(model, items, evaluate, normed)
    elif test_type == "all":
        return test_all(model, evaluate, normed)
    else:
        raise ValueError("unknown test type: ".format(test_type))


def all_test(model, items, evaluate, test_type, normed):
    print("Embedding normalized:{}".format(normed))
    result_one = test_one(model, items, evaluate, normed)
    # result_all = test_all(model, evaluate, normed)
    result_all = None
    print("Finish all test.")
    return result_one, result_all


def print_cover_rate(model):
    r = model.get_cover_rate()
    if r is not None:
        print("cover rate:{:.2f}%".format(100 * r))


def check_model_trainable(model):
    return hasattr(model.mf, "forward_all")


def train(model, optimizer, data_loader, device, normed=False, epoch=None):
    if not check_model_trainable(model):
        print("This model need not be training.")
        return -1
    model.train()
    total_loss = 0
    for i, batch in enumerate(data_loader):
        batch = to_device(batch, device)
        loss = model.get_loss(batch, normed, epoch)
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        total_loss += loss.item()
        # if i%10000 == 0:
        #    print("batch = {} \t logloss={:4f}".format(i, total_loss/(i+1)))
    return total_loss / (i + 1)


config = {}
save_path = "/data/code_yang/cfbe/res/{}/{}/".format(args.out_dir, args.dataset)
if not os.path.exists(save_path):
    print("hasn't {:s}, we creat it first.".format(save_path))
    os.makedirs(save_path)
logger_name = os.path.join(save_path, "out.log")
print("logger:{}".format(logger_name))
logger = get_logger(logger_name)

learning_rate = args.learning_rate  # config['learning_rate']
dataset = args.dataset
embedding_dim = args.embedding_dim
num_negative = args.num_negative
train_type = args.train_type
weight_decay = args.weight_decay
device = torch.device("cuda:{}".format(args.device))
model_name = args.model
test_type = args.test_type
loss_name = args.loss_name
seed = args.out_dir
patience = args.patience
batch_size = args.batch_size
seq_g = args.seq_g
num_layer = args.num_layer
net_dropout = args.net_dropout
emb_dropout = args.emb_dropout
optimizer_name = args.optimizer
sample_alpha = args.sample_alpha
svd = args.svd
bias = args.bias
use_dnn = args.use_dnn
normed = args.normed
noise = args.noise
seed = args.seed

print('model_name', model_name)

matrix_type_mapping = defaultdict(str)
matrix_type_mapping["light"] = "sym"
matrix_type_mapping["mfs"] = "mean"
matrix_type_mapping["smf"] = "mean"
matrix_type_mapping["smfs"] = "mean"
matrix_type = matrix_type_mapping[model_name]
setup_seed(seed)
es = EarlyStopper(patience)
print("dataset={}, learning_rate={}, batch_size={}, test_type={}, patience = {}".format(dataset, learning_rate,
                                                                                        batch_size, test_type,
                                                                                        patience))
print("num_negative={}, train_type={}, embedding_dim={}, matrix_type={}".format(num_negative, train_type, embedding_dim,
                                                                                matrix_type))
print("Start loading train data ...")
start_time = time.time()
num_user, num_item, train_loader, mat, train_mat, test_user_total, test_sample, svd_mat = get_train_new(dataset,
                                                                                                        batch_size,
                                                                                                        num_negative,
                                                                                                        train_type,
                                                                                                        matrix_type,
                                                                                                        svd,
                                                                                                        sample_alpha)

print("{:.2f} seconds, done.".format(time.time() - start_time))
start_time = time.time()
print("Start loading test data ...")
config['num_user'] = num_user
config['num_item'] = num_item
config['embedding_dim'] = embedding_dim
config['seq_g'] = seq_g
config['hidden_units'] = [256]
config['net_dropout'] = net_dropout
config['graph_matrix'] = train_mat
config['num_layer'] = num_layer
config['num_state'] = 5
config['emb_dropout'] = emb_dropout
config['model_name'] = model_name
config['svd_mat'] = svd_mat
config['mat'] = train_mat
config['mat_num_neg'] = num_negative
config['num_neg'] = num_negative
config['bias'] = bias
config['use_dnn'] = use_dnn
config['num_interest'] = 3

model = get_model(model_name, config, loss_name, train_type, weight_decay)
model.name = "topk_{}_{}_batch_{}_dim_{}_{}_neg_{}_{}_{}_norm_{}_emb_dropout{}_dnn_{}".format(args.topk, model.name,
                                                                                              batch_size, embedding_dim,
                                                                                              dataset, num_negative,
                                                                                              optimizer_name,
                                                                                              learning_rate, normed,
                                                                                              emb_dropout, use_dnn)
count_parameters(model)
model.to(device)

if check_model_trainable(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.75, patience=5, verbose=True,
                                                           min_lr=0.000001)
    print("optimizer={}, lr={}, weight_decay={}".format("adam", learning_rate, weight_decay))

evals = Evaluation(mat, train_mat, test_user_total, args.topk)
get_gpu_info()
test_interval = 1
train_loss = -1
print("loss params: {}".format(model.get_loss_params()))
stop_train = False
entropy, aleatoric, epistemic = 0, 0, 0
for epoch_i in range(args.epoch):
    # print_cover_rate(model)

    start_train = time.time()
    train_loss = train(model, optimizer, train_loader, device, normed, epoch_i)
    get_gpu_info()
    print("train loss:{:.4f}, with time {:.2f}s and memory {:.2f}G".format(train_loss, time.time() - start_train,
                                                                           get_memory_info()))
    start_test = time.time()

    result_one, result_all = all_test(model, test_sample, evals, test_type, normed)
    scheduler.step(result_one['recall_one'])

    # entropy, aleatoric, epistemic, _ , _, _= get_uncertainty_new(model, num_user, num_item, embedding_dim, args.mc_sample, device, ana=False)
    # entropy, _ = uncertainty_analy(model)

    print("Finish test, with time {:.2f}s and memory {:.2f}G".format(time.time() - start_test, get_memory_info()))

    # for name, param in model.named_parameters():
    #     if name=='mf.user_std.weight':
    #         user_var = param.data
    #     if name=='mf.item_std.weight':
    #         item_var = param.data

    logger.info(
        "model = {}; epoch = {}; train_loss = {:.6f}; test: logloss_one = {:.6f}; hr_one = {:.6f}; ndcg_one = {:.6f}; entropy={:.4f}; data_noise={:.4f}; model_noise={:.4f}".
            format(model.name + "_one", epoch_i, train_loss, result_one['logloss_one'], result_one['hr_one'],
                   result_one['ndcg_one'], entropy, aleatoric, epistemic))

    if not es.is_continuable(result_one['recall_one'], model):
        stop_train = True

    if stop_train:
        # entropy, aleatoric, epistemic, rand_mat, user_sel, item_sel, entropy, iii = get_uncertainty_new(es.best_model,num_user,num_item,embedding_dim,args.mc_sample,device)

        # predictive_entropy, predictive_aleatoric, predictive_epistemic, predictive_rand_mat = predictive_uncertainty(es.best_model, args.mc_sample, device, test_sample)
        # test_sample_arr = test_sample.detach().cpu()

        # np.save(save_path + 'rand_mat.npy', predictive_rand_mat)
        # np.save(save_path + 'user_sel.npy', user_sel)
        # np.save(save_path + 'item_sel.npy', item_sel)
        # np.save(save_path + 'analytical_entropy.npy', entropy)
        # np.save(save_path + 'analytical_index.npy', iii)
        # np.save(save_path + 'predictive_rand_mat.npy', predictive_rand_mat)
        # np.save(save_path + 'predictive_entropy.npy', predictive_entropy)
        # np.save(save_path + 'predictive_aleatoric.npy', predictive_aleatoric)
        # np.save(save_path + 'predictive_epistemic.npy', predictive_epistemic)

        # np.save(save_path + 'test_sample_arr.npy', test_sample_arr)

        save_weights(es.best_model, save_path + '/' + 'model.pt')

        break

    if epoch_i == (args.epoch - 1):
        # entropy, aleatoric, epistemic, rand_mat, user_sel, item_sel, entropy, iii = get_uncertainty_new(es.best_model, num_user, num_item, embedding_dim,
        #                                                               args.mc_sample,
        #                                                               device)

        # predictive_entropy, predictive_aleatoric, predictive_epistemic, predictive_rand_mat = predictive_uncertainty(es.best_model, args.mc_sample, device, test_sample)
        # test_sample_arr = test_sample.detach().cpu()

        # np.save(save_path+'rand_mat.npy', rand_mat)
        # np.save(save_path+'user_sel.npy', user_sel)
        # np.save(save_path+'item_sel.npy', item_sel)
        # np.save(save_path + 'analytical_entropy.npy', entropy)
        # np.save(save_path + 'analytical_index.npy', iii)
        # np.save(save_path + 'predictive_rand_mat.npy', predictive_rand_mat)
        # np.save(save_path + 'test_sample_arr.npy', test_sample_arr)
        # np.save(save_path + 'predictive_entropy.npy', predictive_entropy)
        # np.save(save_path + 'predictive_aleatoric.npy', predictive_aleatoric)
        # np.save(save_path + 'predictive_epistemic.npy', predictive_epistemic)

        save_weights(es.best_model, save_path + '/' + 'model.pt')







