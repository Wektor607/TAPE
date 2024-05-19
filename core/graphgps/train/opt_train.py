import os
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(dirname(__file__)))))
# standard library imports
import torch
import time
import wandb
import numpy as np
from ogb.linkproppred import Evaluator
from torch_geometric.graphgym.config import cfg
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
from yacs.config import CfgNode as CN

from torch_geometric.data import Data
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling
from graphgps.network.gsaint import GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, GraphSAINTEdgeSampler
import torch.nn.functional as F

# external 
from embedding.tune_utils import param_tune_acc_mrr, mvari_str2csv, save_parmet_tune
from heuristic.eval import get_metric_score
from utils import config_device
from typing import Dict, Tuple
from utils import Logger
# Understand, whu is it work



class Trainer():
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 emb: torch.nn.Module,
                 data: Data,
                 optimizer: torch.optim.Optimizer, 
                 splits: Dict[str, Data], 
                 run: int, 
                 repeat: int,
                 loggers: Logger, 
                 print_logger: None, 
                 device: int):
        
        self.device = device
        self.model = model.to(self.device)
        self.emb = emb
        
        # params
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        self.FILE_PATH = FILE_PATH 
        self.name_tag = cfg.wandb.name_tag
        self.epochs = cfg.train.epochs
        self.batch_size = cfg.train.batch_size
        
        self.test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)
        self.data = data
        self.optimizer = optimizer
        self.loggers = loggers
        self.print_logger = print_logger
        
        model_types = ['VGAE', 'GAE', 'GAT', 'GraphSage', 'GNNStack']
        self.train_func = {model_type: self._train_gae if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._train_vgae for model_type in model_types}
        self.test_func = {model_type: self._test for model_type in model_types}
        self.evaluate_func = {model_type: self._evaluate if model_type in ['GAE', 'GAT', 'GraphSage', 'GNNStack'] else self._evaluate_vgae for model_type in model_types}
        self.evaluator_hit = Evaluator(name='ogbl-collab')
        self.evaluator_mrr = Evaluator(name='ogbl-citation2')
        
        self.run = run
        self.repeat = repeat
        self.results_rank = {}

    def _train_heart(self, 
                     pos_train_weight,
                     device):

        self.model.train()

        train_pos = self.train_data
        total_loss = total_examples = 0

        if self.emb is None: 
            x = self.data.x
            emb_update = 0
        else: 
            x = self.emb.weight
            emb_update = 1

        train_pos = train_pos.t()
        for perm in DataLoader(range(train_pos.size(0)), self.batch_size,
                            shuffle=True):
            self.optimizer.zero_grad()
            num_nodes = x.size(0)

            ######################### remove loss edges from the aggregation
            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
            train_edge_mask = train_pos[mask].transpose(1,0)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)

            # visualize
            if pos_train_weight != None:
                pos_train_weight = pos_train_weight.to(mask.device)
                edge_weight_mask = pos_train_weight[mask]
                edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            else:
                edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)

            # masked adjacency matrix 
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)

            ##################
            # print(adj)
            x = x.to(device)
            adj = adj.to(device)
            h = self.model.encoder(x, adj)

            edge = train_pos[perm].t()
            pos_out = self.model.decoder(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()

            row, col, _ = adj.coo()
            edge_index = torch.stack([col, row], dim=0)
            edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                    num_neg_samples=perm.size(0), method='dense')

            neg_out = self.model.decoder(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

            loss = pos_loss + neg_loss
            loss.backward()

            # delete thies two lines 
            if emb_update == 1: torch.nn.utils.clip_grad_norm_(x, 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            num_examples = perm.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

            return total_loss / total_examples


    def _train_gae(self):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model.encoder(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def _train_vgae(self):
        self.model.train()
        self.optimizer.zero_grad()
        # encoder is VAE, forward is embedding
        # Missed encoder!!!
        z = self.model(self.train_data.x, self.train_data.edge_index)
        loss = self.model.recon_loss(z, self.train_data.pos_edge_label_index)
        loss = loss + (1 / self.train_data.num_nodes) * self.model.kl_loss()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _test(self, data: Data):
        """test"""
        self.model.eval()

        pos_edge_index = self.test_data.pos_edge_label_index
        neg_edge_index = self.test_data.neg_edge_label_index

        z = self.model.encoder(self.test_data.x, self.test_data.edge_index)
        pos_y = z.new_ones(pos_edge_index.size(1)) 
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        return roc_auc_score(y, pred), average_precision_score(y, pred), auc(fpr, tpr)


    @torch.no_grad()
    def _evaluate(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model.encoder(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    @torch.no_grad()
    def _evaluate_vgae(self, data: Data):
       
        self.model.eval()
        pos_edge_index = data.pos_edge_label_index
        neg_edge_index = data.neg_edge_label_index

        z = self.model(data.x, data.edge_index)
        pos_pred = self.model.decoder(z, pos_edge_index)
        neg_pred = self.model.decoder(z, neg_edge_index)
        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        
        hard_thres = (y_pred.max() + y_pred.min())/2

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1)) 
        y = torch.cat([pos_y, neg_y], dim=0)
        
        y_pred[y_pred >= hard_thres] = 1
        y_pred[y_pred < hard_thres] = 0

        acc = torch.sum(y == y_pred)/len(y)
        
        pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
        result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
        result_mrr.update({'acc': round(acc.tolist(), 5)})
    
        return result_mrr
    
    def merge_result_rank(self):
        result_test = self.evaluate_func[self.model_name](self.test_data)
        result_valid = self.evaluate_func[self.model_name](self.valid_data)
        result_train = self.evaluate_func[self.model_name](self.train_data)

        return {
            key: (result_train[key], result_valid[key], result_test[key])
            for key in result_test.keys()
        }
    

    def train(self):  
        best_auc, best_hits, best_hit100 = 0, 0, 0

        for epoch in range(1, self.epochs + 1):
            loss = self.train_func[self.model_name]()

            # if epoch % 100 == 0:
            if epoch % 10 == 0:
                self.results_rank = self.merge_result_rank()
                self.print_logger.info(self.results_rank)
                
                # for key, result in results_rank.items():
                #     # result - (train, valid, test)
                #     self.loggers[key].add_result(self.run, result)
                    # print(self.loggers[key].results)
                    
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_train: {loss:.4f}, AUC: {self.results_rank["AUC"][0]:.4f}, AP: {self.results_rank["AP"][0]:.4f}, MRR: {self.results_rank["MRR"][0]:.4f}, Hit@10 {self.results_rank["Hits@10"][0]:.4f}')
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_valid: {loss:.4f}, AUC: {self.results_rank["AUC"][1]:.4f}, AP: {self.results_rank["AP"][1]:.4f}, MRR: {self.results_rank["MRR"][1]:.4f}, Hit@10 {self.results_rank["Hits@10"][1]:.4f}')               
                self.print_logger.info(f'Epoch: {epoch:03d}, Loss_test: {loss:.4f}, AUC: {self.results_rank["AUC"][2]:.4f}, AP: {self.results_rank["AP"][2]:.4f}, MRR: {self.results_rank["MRR"][2]:.4f}, Hit@10 {self.results_rank["Hits@10"][2]:.4f}')               
                    
                    
                for key, result in self.results_rank.items():
                    self.loggers[key].add_result(self.run, result)
                    if epoch % 500 == 0:
                        for key, result in self.results_rank.items():
                            train_hits, valid_hits, test_hits = result
                            self.print_logger.info(
                                f'Run: {self.run + 1:02d}, '
                                f'Key: {key}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                        self.print_logger.info('---')
                        

        return best_auc, best_hits



    def result_statistic(self):
        result_all_run = {}
        for key in self.loggers:
            print(key)
            best_metric,  best_valid_mean, mean_list, var_list = self.loggers[key].calc_all_stats()
            
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

            elif key == 'Hits@100':
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]

        print(f'{best_metric_valid_str} {best_auc_valid_str}')
        print(best_metric_valid_str)
        best_auc_metric = best_valid_mean_metric
        return best_valid_mean_metric, best_auc_metric, result_all_run
    

    def save_result(self, results_dict: Dict[str, float]):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, f'{self.data_name}_wb_acc_mrr.csv')
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        
       
        mvari_str2csv(self.name_tag, results_dict, acc_file)


    def save_tune(self, results_dict: Dict[str, float], to_file):  # sourcery skip: avoid-builtin-shadow
        
        root = os.path.join(self.FILE_PATH, cfg.out_dir)
        acc_file = os.path.join(root, to_file)
        self.print_logger.info(f"save to {acc_file}")
        os.makedirs(root, exist_ok=True)
        
        save_parmet_tune(self.name_tag, results_dict, acc_file)


class Trainer_Saint(Trainer):
    def __init__(self, 
                 FILE_PATH: str, 
                 cfg: CN, 
                 model: torch.nn.Module, 
                 emb: torch.nn.Module,
                 data: Data,
                 optimizer: torch.optim.Optimizer, 
                 splits: Dict[str, Data], 
                 run: int, 
                 repeat: int,
                 loggers: Logger, 
                 print_logger: None,  # Ensure this is correctly defined and passed
                 device: torch.device,
                 gsaint=None,
                 batch_size_sampler=None, 
                 walk_length=None, 
                 num_steps=None, 
                 sample_coverage=None):
        # Correctly pass all parameters expected by the superclass constructor
        super().__init__(FILE_PATH, cfg, model, emb, data, optimizer, splits, run, repeat, loggers, print_logger, device)
        
        self.device = device 
        self.print_logger = print_logger                
        self.model = model.to(self.device)
        self.emb = emb
        self.data = data.to(self.device)
        self.model_name = cfg.model.type 
        self.data_name = cfg.data.name
        
        self.FILE_PATH = FILE_PATH 
        self.epochs = cfg.train.epochs
        
        # GSAINT splitting
        if gsaint is not None:
            device_cpu = torch.device('cpu')
            self.test_data  = GraphSAINTRandomWalkSampler(splits['test'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
            self.train_data = GraphSAINTRandomWalkSampler(splits['train'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
            self.valid_data = GraphSAINTRandomWalkSampler(splits['valid'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
        else:
            self.test_data  = splits['test'].to(self.device)
            self.train_data = splits['train'].to(self.device)
            self.valid_data = splits['valid'].to(self.device)

        self.optimizer  = optimizer
    
    def global_to_local(self, edge_label_index, node_idx):

        # Make dict where key: local indexes, value: global indexes
        global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(node_idx.tolist())}

        # Create new local edge indexes
        edge_indices = [
            torch.tensor([global_to_local.get(idx.item(), -1) for idx in edge_label_index[0]], dtype=torch.long),
            torch.tensor([global_to_local.get(idx.item(), -1) for idx in edge_label_index[1]], dtype=torch.long)
        ]

        local_indices = torch.stack(edge_indices, dim=0)

        # Since we are going through the entire list of positive/negative indices, 
        # some edges in the subgraph will be marked -1, so we delete them
        valid_indices = (local_indices >= 0).all(dim=0)
        local_indices = local_indices[:, valid_indices]

        return local_indices
    
    def _train_gae(self):
        self.model.train()
        total_loss = total_examples = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()
            subgraph = subgraph.to(self.device)

            z = self.model.encoder(subgraph.x, subgraph.edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.node_index)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.node_index)

            loss = self.model.recon_loss(z, local_pos_indices, local_neg_indices)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
            total_examples += subgraph.num_nodes
        
        return total_loss / total_examples
    
    def _train_vgae(self):
        self.model.train()
        total_loss = total_examples = 0
        for subgraph in self.train_data:
            self.optimizer.zero_grad()
            subgraph = subgraph.to(self.device)

            z = self.model(subgraph.x, subgraph.edge_index)

            local_pos_indices = self.global_to_local(subgraph.pos_edge_label_index, subgraph.node_index)
            local_neg_indices = self.global_to_local(subgraph.neg_edge_label_index, subgraph.node_index)

            loss = self.model.recon_loss(z, local_pos_indices, local_neg_indices)
            loss += (1 / subgraph.num_nodes) * self.model.kl_loss()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * subgraph.num_nodes
            total_examples += subgraph.num_nodes
        
        return total_loss / total_examples

    @torch.no_grad()
    def _evaluate(self, data_loader: Data):
        self.model.eval()
        accumulated_metrics = []

        for data in data_loader:
            data = data.to(self.device)

            local_pos_indices = self.global_to_local(data.pos_edge_label_index, data.node_index)
            local_neg_indices = self.global_to_local(data.neg_edge_label_index, data.node_index)
            
            z = self.model.encoder(data.x, data.edge_index)
            pos_pred = self.model.decoder(z, local_pos_indices)
            neg_pred = self.model.decoder(z, local_neg_indices)
            y_pred = torch.cat([pos_pred, neg_pred], dim=0)

            hard_thres = (y_pred.max() + y_pred.min())/2

            pos_y = z.new_ones(local_pos_indices.size(1))
            neg_y = z.new_zeros(local_neg_indices.size(1)) 
            y = torch.cat([pos_y, neg_y], dim=0)
            
            y_pred[y_pred >= hard_thres] = 1
            y_pred[y_pred < hard_thres] = 0
            acc = torch.sum(y == y_pred) / len(y)

            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
            result_mrr.update({'acc': round(acc.item(), 5)})
            accumulated_metrics.append(result_mrr)

        # Aggregate results from accumulated_metrics
        aggregated_results = {}
        for result in accumulated_metrics:
            for key, value in result.items():
                if key in aggregated_results:
                    aggregated_results[key].append(value)
                else:
                    aggregated_results[key] = [value]

        # Calculate average results
        averaged_results = {key: sum(values) / len(values) for key, values in aggregated_results.items()}

        return averaged_results

    @torch.no_grad()
    def _evaluate_vgae(self, data_loader):
        self.model.eval()
        accumulated_metrics = []

        for data in data_loader:
            data = data.to(self.device)

            local_pos_indices = self.global_to_local(data.pos_edge_label_index, data.node_index)
            local_neg_indices = self.global_to_local(data.neg_edge_label_index, data.node_index)
            
            z = self.model(data.x, data.edge_index)
            pos_pred = self.model.decoder(z, local_pos_indices)
            neg_pred = self.model.decoder(z, local_neg_indices)
            y_pred = torch.cat([pos_pred, neg_pred], dim=0)

            hard_thres = (y_pred.max() + y_pred.min())/2

            pos_y = z.new_ones(local_pos_indices.size(1))
            neg_y = z.new_zeros(local_neg_indices.size(1)) 
            y = torch.cat([pos_y, neg_y], dim=0)
            
            y_pred[y_pred >= hard_thres] = 1
            y_pred[y_pred < hard_thres] = 0
            acc = torch.sum(y == y_pred) / len(y)
            
            pos_pred, neg_pred = pos_pred.cpu(), neg_pred.cpu()
            result_mrr = get_metric_score(self.evaluator_hit, self.evaluator_mrr, pos_pred, neg_pred)
            result_mrr.update({'acc': round(acc.item(), 5)})
            accumulated_metrics.append(result_mrr)

        # Aggregate results from accumulated_metrics
        aggregated_results = {}
        for result in accumulated_metrics:
            for key, value in result.items():
                if key in aggregated_results:
                    aggregated_results[key].append(value)
                else:
                    aggregated_results[key] = [value]

        # Calculate average results
        averaged_results = {key: sum(values) / len(values) for key, values in aggregated_results.items()}

        return averaged_results

