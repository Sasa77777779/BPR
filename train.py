import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

device = 'cpu'


def LoadData():
    train_data = pd.read_csv('./train_data', delim_whitespace=True,
                             header=None, names=['user', 'item_i', 'item_j'], usecols=['user', 'item_i', 'item_j'],
                             dtype={'user': np.int32, 'item_i': np.int32, 'item_j': np.int32})

    test_data = pd.read_csv('./test_data', delim_whitespace=True,
                            header=None, names=['user', 'item'], usecols=['user', 'item'],
                            dtype={'user': np.int32, 'item': np.int32})

    train_history = pd.read_csv('./train_history.txt', delim_whitespace=True,
                                header=None, names=['user', 'item'], usecols=['user', 'item'],
                                dtype={'user': np.int32, 'item_i': np.int32, 'item_j': np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item_i'].max() + 1

    train_data = train_data.values.tolist()
    test_data = test_data.values.tolist()
    train_history = train_history.values.tolist()

    return train_data, test_data, user_num, item_num, train_history


class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(BPR, self).__init__()

        # 用户矩阵和物品矩阵的定义
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        # 用户矩阵和物品矩阵的初始化
        # nn.init.normal_
        # 从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    # user,item_i,item_j都是矩阵
    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        # 得到item矩阵中每个物品的得分
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)

        return prediction_i, prediction_j


def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def evaluate(model, test_loader, top_k):
    Hit = []
    Ndcg = []

    for u, i in test_loader:
        prediction_i, _ = model(u.to(device=device), i.to(device=device), i.to(device=device))
        _, index = torch.topk(prediction_i, top_k)
        recommends = torch.take(i.to(device=device), index).tolist()

        positive_item = i[0].item()

        Hit.append(hit(positive_item, recommends))
        Ndcg.append(ndcg(positive_item, recommends))

    return np.mean(Hit), np.mean(Ndcg)


class MyDataset(Dataset):
    def __init__(self, history, item_number):
        super().__init__()
        self.history = history
        self.history_set = {(x, y) for x, y in history}
        self.item_number = item_number

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        u, i = self.history[index]
        j = np.random.randint(self.item_number)
        while (u, j) in self.history_set:
            j = np.random.randint(self.item_number)
        return u, i, j


if __name__ == '__main__':
    lr = 0.01
    lamda = 0.001
    batch_size = 4096
    epochs = 5
    top_k = 10
    factor_num = 16

    cudnn.benchmark = True

    train_data, test_data, user_num, item_num, train_history = LoadData()

    train_loader = data.DataLoader(MyDataset(train_history, item_num), batch_size=batch_size,
                                   shuffle=True, num_workers=4)
    test_loader = data.DataLoader(test_data, batch_size=100,
                                  shuffle=False, num_workers=0)

    model = BPR(user_num, item_num, factor_num)
    model = model.to(device=device)

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamda)

    for epoch in range(epochs):
        model.train()

        for user, item_i, item_j in tqdm(train_loader):
            model.zero_grad()
            prediction_i, prediction_j = model(user.to(device=device), item_i.to(device=device),
                                               item_j.to(device=device))
            loss = -(prediction_i - prediction_j).sigmoid().log().sum()
            loss.backward()
            optimizer.step()

        model.eval()
        HR, NDCG = evaluate(model, test_loader, top_k)
        # np.mean计算均值
        print("HR:{:.3}\tNDCG:{:.3}".format(np.mean(HR), np.mean(NDCG)))
