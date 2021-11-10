import numpy as np
import pandas as pd
from tqdm import tqdm


def Readfile():
    history_data = pd.read_csv('./ml-100k/u.data', delim_whitespace=True,
                               header=None, names=['user', 'item'], usecols=['user', 'item'],
                               dtype={'user': np.int32, 'item': np.int32})

    user_num = history_data['user'].max()
    item_num = history_data['item'].max()

    history_data = history_data.values.tolist()

    history_data.sort()

    train_history = []
    test_history = []
    count = 0
    for u, i in history_data:
        if u != count:
            test_history.append((u, i))
            count = count + 1
        else:
            train_history.append((u, i))

    return history_data, train_history, test_history, user_num, item_num


def Sampling(history_data, train_history, test_history, item_num, train_sample_num, test_sample_num):
    train_data = []
    test_data = []
    history_data_set = {(x, y) for x, y in history_data}
    for u, i in tqdm(train_history, desc="train_history"):
        for t in range(train_sample_num):
            j = np.random.randint(item_num)
            while (u, j) in history_data_set:
                j = np.random.randint(item_num)
            train_data.append((u, i, j))

    for u, i in tqdm(test_history, desc="test_history"):
        test_data.append((u, i))
        for t in range(test_sample_num):
            j = np.random.randint(item_num)
            while (u, j) in history_data_set:
                j = np.random.randint(item_num)
            test_data.append((u, j))

    return train_data, test_data


if __name__ == '__main__':
    train_sample_num = 5
    test_sample_num = 99

    history_data, train_history, test_history, user_num, item_num = Readfile()
    train_data, test_data = Sampling(history_data, train_history, test_history, item_num, train_sample_num,
                                     test_sample_num)

    fp1 = open('./train_data', 'w')
    fp2 = open('./test_data', 'w')

    for u, i, j in train_data:
        fp1.write(str(u) + '\t' + str(i) + '\t' + str(j) + '\n')

    for u, i in test_data:
        fp2.write(str(u) + '\t' + str(i) + '\n')

    with open("./train_history.txt", "w") as fp:
        for u, i in train_history:
            fp.write(f"{u}\t{i}\n")
