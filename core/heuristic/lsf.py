"""
A selection of heuristic methods (Personalized PageRank, Adamic Adar and Common Neighbours) for link prediction
"""

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
np.seterr(divide = 'ignore') 

def CN(A, edge_index, batch_size=100000):
    """
    Common neighbours
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    edge_index = edge_index.t()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def InverseRA(A, edge_index, batch_size=100000):
    
    """
    Inverse Adamic Adar
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    edge_index = edge_index.t()
    multiplier = np.exp((A.sum(axis=0)))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated InverseRA for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def AA(A, edge_index, batch_size=100000):
    """
    Adamic Adar
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    edge_index = edge_index.t()
    multiplier = 1 / (np.log(A.sum(axis=0)))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Adamic Adar for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def RA(A, edge_index, batch_size=100000):
    """
    Resource Allocation https://arxiv.org/pdf/0901.0553.pdf
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    edge_index = edge_index.t()
    multiplier = 1 / A.sum(axis=0)
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Resource Allocation for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


