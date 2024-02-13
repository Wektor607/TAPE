import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize
import json
from torch_geometric.transforms import RandomLinkSplit
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from lpda.adjacency import plot_coo_matrix
from data_utils.load_pubmed import get_raw_text_pubmed, get_pubmed_casestudy, parse_pubmed
import matplotlib.pyplot as plt
from lpda.adjacency import construct_sparse_adj
import scipy.sparse as ssp

from heuristic.lsf import CN, AA, RA, InverseRA
from heuristic.gsf import Ben_PPR, shortest_path, katz_apro, katz_close , SymPPR
from typing import Dict
from torch_geometric.data import Dataset
import torch
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from heuristic.eval import evaluate_auc, evaluate_hits, evaluate_mrr, get_metric_score, get_prediction
from utils import get_git_repo_root_path, append_acc_to_excel, append_mrr_to_excel
from heuristic.semantic_similarity import pairwise_prediction



FILE_PATH = get_git_repo_root_path() + '/'


def get_pubmed_casestudy(config):
    corrected = False
    undirected = config.data.undirected
    include_negatives = config.data.include_negatives
    val_pct = config.data.val_pct
    test_pct = config.data.test_pct
    split_labels = config.data.split_labels
    
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('./dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    x = torch.tensor(data_X)
    edge_index = torch.tensor(data_edges)
    y = torch.tensor(data_Y)
    num_nodes = data.num_nodes
    
    # split data
    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        train_id = [i for i in train_id if not is_mistake[i]]
        val_id = [i for i in val_id if not is_mistake[i]]
        test_id = [i for i in test_id if not is_mistake[i]]


    data = Data(x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
        node_attrs=x, 
        edge_attrs = None, 
        graph_attrs = None
    )        

    undirected = data.is_undirected()

    transform = RandomLinkSplit(is_undirected=undirected, num_val=val_pct, num_test=test_pct,
                                add_negative_train_samples=include_negatives, split_labels=split_labels)
    train_data, val_data, test_data = transform(dataset._data)
    splits = {'train': train_data, 'valid': val_data, 'test': test_data}
    
    dataset._data = data
    
    return dataset, data_pubid, splits

       
def eval_pubmed_acc(name) -> Dict:
    dataset, data_pubid, splits = get_pubmed_casestudy(corrected=False, SEED=0)
    print(dataset)

    test_split = splits['test']
    labels = test_split.edge_label
    test_index = test_split.edge_label_index
    
    edge_index = splits['test'].edge_index
    edge_weight = torch.ones(edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    result_acc = {}
    for use_lsf in ['CN', 'AA', 'RA', 'InverseRA']:
        scores, edge_index = eval(use_lsf)(A, test_index)
        
        plt.figure()
        plt.plot(scores)
        plt.plot(labels)
        plt.savefig(f'pubmed{use_lsf}.png')
        
        acc = torch.sum(scores == labels)/scores.shape[0]
        result_acc.update({f"{use_lsf}_acc" :acc})
        
    m = construct_sparse_adj(edge_index)
    plot_coo_matrix(m, f'{name}_test_edge_index.png')
            
    # 'shortest_path', 'katz_apro', 'katz_close', 'Ben_PPR'
    for use_gsf in ['Ben_PPR', 'SymPPR']:
        scores, edge_reindex = eval(use_gsf)(A, test_index)
        
        # print(scores)
        # print(f" {use_heuristic}: accuracy: {scores}")
        pred = torch.zeros(scores.shape)
        cutoff = 0.05
        thres = scores.max()*cutoff 
        pred[scores <= thres] = 0
        pred[scores > thres] = 1

        acc = torch.sum(pred == labels)/scores.shape[0]
        result_acc.update({f"{use_gsf}_acc" :acc})
    
    
    for use_gsf in ['shortest_path', 'katz_apro', 'katz_close']:
        scores = eval(use_gsf)(A, test_index)
        
        pred = torch.zeros(scores.shape)
        thres = scores.min()*10
        pred[scores <= thres] = 0
        pred[scores > thres] = 1
        
        acc = torch.sum(pred == labels)/labels.shape[0]
        result_acc.update({f"{use_gsf}_acc" :acc})
        
    for use_heuristic in ['pairwise_pred']:
        for dist in ['dot']:
            scores = pairwise_prediction(dataset._data.x, test_index, dist)
            test_pred = torch.zeros(scores.shape)
            cutoff = 0.25
            thres = scores.max()*cutoff 
            test_pred[scores <= thres] = 0
            test_pred[scores > thres] = 1
            acc = torch.sum(test_pred == labels)/labels.shape[0]
            
            plt.figure()
            plt.plot(test_pred)
            plt.plot(labels)
            plt.savefig(f'{use_heuristic}.png')
        
        result_acc.update({f"{use_heuristic}_acc" :acc})
        
    return result_acc

def eval_pubmed_mrr(name):
    
    dataset, data_pubid, splits = get_pubmed_casestudy(corrected=False, SEED=0,
                            undirected = True,
                            include_negatives = True,
                            val_pct = 0.15,
                            test_pct = 0.05,
                            split_labels = True)
    print(dataset._data)

    full_edge_index = splits['test'].edge_index
    full_edge_weight = torch.ones(full_edge_index.size(1))
    num_nodes = dataset._data.num_nodes
    
    m = construct_sparse_adj(full_edge_index)
    plot_coo_matrix(m, f'{name}_test_edge_index.png')
    
    full_A = ssp.csr_matrix((full_edge_weight.view(-1), (full_edge_index[0], full_edge_index[1])), shape=(num_nodes, num_nodes)) 

    # only for debug
    pos_test_index = splits['test'].pos_edge_label_index
    neg_test_index = splits['test'].neg_edge_label_index
    
    pos_m = construct_sparse_adj(pos_test_index)
    plot_coo_matrix(pos_m, f'{name}_test_pos_index.png')
    neg_m = construct_sparse_adj(neg_test_index)
    plot_coo_matrix(neg_m, f'{name}_test_neg_index.png')
    
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    result_mrr = {}
    # 'InverseRA'
    for use_heuristic in ['CN', 'AA', 'RA']:
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
        
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_mrr.update({f'{use_heuristic}': result})

    # , 'SymPPR'
    for use_heuristic in ['Ben_PPR']:
        pos_test_pred, _ = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred, _ = eval(use_heuristic)(full_A, neg_test_index)
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_mrr.update({f'{use_heuristic}': result})
    
    
    for use_heuristic in ['shortest_path', 'katz_apro', 'katz_close']:
        pos_test_pred = eval(use_heuristic)(full_A, pos_test_index)
        neg_test_pred = eval(use_heuristic)(full_A, neg_test_index)
        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
        result_mrr.update({f'{use_heuristic}': result})

    for use_heuristic in ['pairwise_pred']:
        for dist in ['dot']:
            pos_test_pred = pairwise_prediction(dataset._data.x, pos_test_index, dist)
            neg_test_pred = pairwise_prediction(dataset._data.x, neg_test_index, dist)
            result = get_metric_score(evaluator_hit, evaluator_mrr, pos_test_pred, neg_test_pred)
            result_mrr.update({f'{use_heuristic}_{dist}': result})
            
    return result_mrr


if __name__ == "__main__":
    name = 'pubmed'
    result_acc = eval_pubmed_acc(name)
    result_mrr = eval_pubmed_mrr(name)
    root = FILE_PATH + 'results'
    acc_file = root + f'/{name}_acc.csv'
    mrr_file = root + f'/{name}_mrr.csv'
    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)
    
    # TEST CODE 
    append_acc_to_excel(result_acc, acc_file, name)
    append_mrr_to_excel(result_mrr, mrr_file)
    