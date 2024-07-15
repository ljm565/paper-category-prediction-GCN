import random
import pandas as pd
from scipy.linalg import fractional_matrix_power

import torch



def load_graph_data(cite_data_path, content_data_path):
    # load edge information
    cites = pd.read_csv(cite_data_path, sep='\t', header=None, names=['target', 'source'])
    
    # load node feature, the last column is label
    col = ['word_{}'.format(i) for i in range(1433)] + ['category']  # 1433 is the number of words in dict and final category
    contents = pd.read_csv(content_data_path, delimiter='\t', header=None, names=col)

    # make label and node location dictionaries
    category = {c: i for i, c in enumerate(sorted(list(set(contents['category']))))}   # sort: for reproduction
    node_loc = {n: i for i, n in enumerate(list(contents.index))}

    return cites, contents, category, node_loc


def make_gnn_data(cite_data_path, content_data_path, dynamic=True, directed=False):
    cites, contents, category, node_loc = load_graph_data(cite_data_path, content_data_path)
    
    # adjacency matrix
    data_len = len(contents)
    adj = torch.zeros((data_len, data_len))
    adj.fill_diagonal_(1)
    for row in torch.tensor(cites.values):
        x, y = row[0].item(), row[1].item()
        if directed == False:
            adj[node_loc[x], node_loc[y]] = 1
        adj[node_loc[y], node_loc[x]] = 1
    adj = matrix_normalize(adj, dynamic)

    # feature matrix
    feature = matrix_normalize(torch.FloatTensor(contents.iloc[:, :-1].values), dynamic=False)

    # label matrix
    label = torch.LongTensor([category[c] for c in contents.iloc[:, -1]])

    return adj, feature, label


def matrix_normalize(x, dynamic):
    """
    dynamic True: D^-0.5 A D^-0.5
    dynamic False: D^-1 A
    """
    if dynamic:
        degree = torch.sum(x, dim=1)
        D_inv = torch.FloatTensor(fractional_matrix_power(torch.diag(degree), -0.5))
        x = torch.mm(torch.mm(D_inv, x), D_inv)
    else:
        degree = torch.sum(x, dim=1)
        D_inv = torch.linalg.inv(torch.diag(degree))
        x = torch.mm(D_inv, x)
    return x


def split_data(data_len):
    all_idx = list(range(data_len))

    train_len = int(data_len * 0.7)
    val_len = int(data_len * 0.15)

    train_idx = random.sample(all_idx, train_len)
    tmp_idx = list(set(all_idx) - set(train_idx))
    val_idx = random.sample(tmp_idx, val_len)
    test_idx = list(set(tmp_idx) - set(val_idx))

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    return train_idx, val_idx, test_idx