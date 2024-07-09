import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pandas as pd
from graphgps.utility.utils import get_git_repo_root_path
from typing import Dict
import numpy as np
import scipy.sparse as ssp
import json
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from yacs.config import CfgNode as CN
from data_utils.dataset import CustomLinkDataset
from data_utils.load_data_nc import load_tag_cora, load_tag_pubmed, \
    load_tag_product, load_tag_ogbn_arxiv, load_tag_product, \
    load_tag_arxiv23, load_graph_cora, load_graph_pubmed, \
    load_graph_arxiv23, load_graph_ogbn_arxiv, load_text_cora, \
    load_text_pubmed, load_text_arxiv23, load_text_ogbn_arxiv, \
    load_text_product, load_text_citeseer, load_text_citationv8, \
    load_graph_citeseer, load_graph_citationv8, load_graph_pwc_large, load_text_pwc_large, \
    load_graph_pwc_medium, load_text_pwc_medium, load_text_pwc_small,  load_graph_pwc_small
from graphgps.utility.utils import get_git_repo_root_path, config_device, init_cfg_test
from typing import Dict, Tuple, List, Union
import torch
from lpda.lcc_3 import find_scc_direc, use_lcc_direc

FILE = 'core/dataset/ogbn_products_orig/ogbn-products.csv'
FILE_PATH = get_git_repo_root_path() + '/'


# arxiv_2023
def load_taglp_arxiv2023(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_arxiv23()
    if data.is_directed() is True:
        data.edge_index = to_undirected(data.edge_index)
        undirected = True
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taglp_cora(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, data_citeid = load_graph_cora(False)
    text = load_text_cora(data_citeid)
    # text = None
    undirected = data.is_undirected()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taglp_ogbn_arxiv(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_ogbn_arxiv(False)
    text = load_text_ogbn_arxiv()
    undirected = data.is_undirected()

    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data

def load_taglp_pwc_large(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_pwc_large()
    text = load_text_pwc_large()
    undirected = data.is_undirected()

    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def get_edge_split(data: Data,
                   undirected: bool,
                   device: Union[str, int],
                   val_pct: float,
                   test_pct: float,
                   include_negatives: bool,
                   split_labels: bool):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        RandomLinkSplit(is_undirected=undirected,
                        num_val=val_pct,
                        num_test=test_pct,
                        add_negative_train_samples=include_negatives,
                        split_labels=split_labels),

    ])
    del data.adj_t, data.e_id, data.batch_size, data.n_asin, data.n_id
    train_data, val_data, test_data = transform(data)
    return {'train': train_data, 'valid': val_data, 'test': test_data}


def load_taglp_product(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data, text = load_tag_product()
    undirected = data.is_undirected()

    cfg = config_device(cfg)

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taglp_pubmed(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_pubmed(False)
    text = load_text_pubmed()
    undirected = data.is_undirected()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data

def load_taglp_citeseer(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument

    data = load_graph_citeseer()
    text = load_text_citeseer()
    undirected = data.is_undirected()

    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data

def load_taglp_citationv8(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    # add one default argument
    
    data = load_graph_citationv8()
    text = load_text_citationv8()
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


 
def load_taplp_pwc_large(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    data = load_graph_pwc_large()
    text = load_text_pwc_large()
    
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data

def load_taplp_pwc_medium(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    data = load_graph_pwc_medium()
    text = load_text_pwc_medium()
    
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data


def load_taplp_pwc_small(cfg: CN) -> Tuple[Dict[str, Data], List[str]]:
    data = load_graph_pwc_small() 
    text = load_text_pwc_small()
    
    if data.is_directed() is True:
        data.edge_index  = to_undirected(data.edge_index)
        undirected  = True 
    else:
        undirected = data.is_undirected()
        
    splits = get_edge_split(data,
                            undirected,
                            cfg.device,
                            cfg.split_index[1],
                            cfg.split_index[2],
                            cfg.include_negatives,
                            cfg.split_labels
                            )
    return splits, text, data




# TEST CODE
if __name__ == '__main__':
    args = init_cfg_test()

    print('pwc_large')
    print(args.data)
    splits, text, data = load_taglp_pwc_large(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")

    from lpda.lcc_3 import use_lcc
    # return the largest connected components with text attrs
    graph = torch.load(FILE_PATH+'core/dataset/pwc_large/pwc_w2v_large_undir.pt')
    data_lcc = use_lcc(graph)
    root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/'
    torch.save(data_lcc, root+'core/dataset/pwc_medium/pwc_w2v_medium_undir.pt')
    
    exit(-1)
    # large undirected graph to medium undirected graph
    # w2v
    text = load_text_pwc_large()
    graph = torch.load(FILE_PATH + 'core/dataset/pwc_large/pwc_w2v_large_undir.pt')
    
    largest_scc = find_scc_direc(graph)
    # save id
    # Print the nodes in the largest SCC
    print("Nodes in the largest strongly connected component:", len(largest_scc))
    
    # Extract the subgraph corresponding to the largest SCC
    subgraph = use_lcc_direc(data, largest_scc)
    
    # Print the subgraph details
    print("Subgraph edge index:", subgraph.edge_index)
    print("Subgraph node features:", subgraph.x)
    
    # save graph
    root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/dataset/pwc_medium/'
    torch.save(subgraph, root+'pwc_w2v_medium_undir.pt')
    # save text
    # TODO
    
    
    # large undirected graph to medium undirected graph
    # tfidf
    graph = torch.load(FILE_PATH + 'core/dataset/pwc_large/pwc_tfidf_large_undir.pt')
    
    largest_scc = find_scc_direc(graph)
    
    # Print the nodes in the largest SCC
    print("Nodes in the largest strongly connected component:", len(largest_scc))
    
    # Extract the subgraph corresponding to the largest SCC
    subgraph = use_lcc_direc(data, largest_scc)
    
    # Print the subgraph details
    print("Subgraph edge index:", subgraph.edge_index)
    print("Subgraph node features:", subgraph.x)

    root = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/dataset/pwc_medium/'
    torch.save(subgraph, root+'pwc_w2v_medium_undir.pt')
    # save text
    #TODO
    
    exit(-1)
    
    print('arxiv2023')
    splits, text, data  = load_taglp_arxiv2023(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")
    
    print('citationv8')
    splits, text, data = load_taglp_citationv8(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")

    print('cora')
    splits, text, data = load_taglp_cora(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")
    
    print('pubmed')
    splits, text, data = load_taglp_pubmed(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")


    
    print(args.data)
    splits, text, data = load_taglp_ogbn_arxiv(args.data)
    print(f'directed: {data.is_directed()}')
    print(data)
    print(text[0])
    print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")


    # print('product')
    # splits, text, data = load_taglp_product(args.data)
    # print(f'directed: {data.is_directed()}')
    # print(data)
    # print(text[0])
    # print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    # print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    # print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")

    # splits, text, data = load_taglp_citeseer(args.data)
    # print(f'directed: {data.is_directed()}')
    # print(data)
    # # print(text[0])
    # print(f"train dataset: {splits['train'].pos_edge_label.shape[0]*2} edges.")
    # print(f"valid dataset: {splits['valid'].pos_edge_label.shape[0]*2} edges.")
    # print(f"test dataset: {splits['test'].pos_edge_label.shape[0]*2} edges.")