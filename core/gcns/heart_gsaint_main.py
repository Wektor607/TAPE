import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gcns.heart_main import *
from gcns.gsaint_main import get_loader_RW
import itertools
from tqdm import tqdm

if __name__ == "__main__":

    FILE_PATH = get_git_repo_root_path() + '/'

    args = parse_args()
    # Load args file
    
    cfg = set_cfg(FILE_PATH, args)
    cfg.merge_from_list(args.opts)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    device = config_device(cfg)

    cfg.train.device = device
    print(f"device {device}")
    
    hyperparameter_search = {
        'out_channels'      : [32, 64],
        'hidden_channels'   : [32],
        'lr'                : [0.1, 0.01, 0.001, 0.0001],
        'batch_size'        : [32, 1024, 2048, 4096],
        'batch_size_sampler': [32, 64, 128, 256],
        'walk_length'       : [10, 20, 50, 100, 150, 200],
        'num_steps'         : [10, 20, 30],
        'sample_coverage'   : [50, 100, 150, 200, 250]
    }

    if cfg.model.sampler == 'gsaint':
        sampler = get_loader_RW
    else:
        sampler = None 
    
    for out, hidden, lr, batch_size, batch_size_sampler, walk_length, num_steps, sample_coverage in tqdm(itertools.product(*hyperparameter_search.values())):
        cfg.model.out_channels = out
        cfg.model.hidden_channels = hidden
        cfg.train.lr = lr
        cfg.train.batch_size = batch_size

        dataset, splits, emb, cfg, train_edge_weight = data_preprocess(cfg)

        pprint(cfg)
        model = eval(cfg.model.type)(cfg.model.input_channels, cfg.model.hidden_channels,
                                    cfg.model.hidden_channels, cfg.model.num_layers, 
                                    cfg.model.dropout).to(device)
        
        score_func = eval(cfg.score_model.name)(cfg.score_model.hidden_channels, 
                                                cfg.score_model.hidden_channels,
                                                1, 
                                                cfg.score_model.num_layers_predictor, 
                                                cfg.score_model.dropout).to(device)

        # train_pos = data['train_pos'].to(x.device)

        # eval_metric = args.metric
        evaluator_hit = Evaluator(name='ogbl-collab')
        evaluator_mrr = Evaluator(name='ogbl-citation2')

        # config reset parameters 
        model.reset_parameters()
        score_func.reset_parameters()
        
        if cfg.model.emb is True:
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()),lr=cfg.train.lr, weight_decay=cfg.train.l2)

        if cfg.model.sampler is not None:
            device_cpu = torch.device('cpu')
            test_data = sampler(splits['test'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
            train_data = sampler(splits['train'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
            valid_data = sampler(splits['valid'].to(device_cpu), batch_size_sampler, walk_length, num_steps, sample_coverage)
        else:
            test_data = splits['test']
            train_data = splits['train']
            valid_data = splits['valid']
        
        test_data = splits['test'].to(self.device)
        self.train_data = splits['train'].to(self.device)
        self.valid_data = splits['valid'].to(self.device)

        if cfg.data.name =='ogbl-collab':
            eval_metric = 'Hits@50'
        elif cfg.data.name =='ogbl-ddi':
            eval_metric = 'Hits@20'

        elif cfg.data.name =='ogbl-ppa':
            eval_metric = 'Hits@100'
        
        elif cfg.data.name =='ogbl-citation2':
            eval_metric = 'MRR'
            
        elif cfg.data.name in ['cora', 'pubmed', 'arxiv_2023']:
            eval_metric = 'Hits@100'
            
        if cfg.data.name != 'ogbl-citation2':
            pos_train_edge = splits['train'].edge_index

            pos_valid_edge = splits['valid'].pos_edge_label_index
            neg_valid_edge = splits['valid'].neg_edge_label_index
            pos_test_edge = splits['test'].pos_edge_label_index
            neg_test_edge = splits['test'].neg_edge_label_index
        
        else:
            source_edge, target_edge = splits['train']['source_node'], splits['train']['target_node']
            pos_train_edge = torch.cat([source_edge.unsqueeze(0), target_edge.unsqueeze(0)], dim=0)

            # idx = torch.randperm(split_edge['train']['source_node'].numel())[:split_edge['valid']['source_node'].size(0)]
            # source, target = split_edge['train']['source_node'][idx], split_edge['train']['target_node'][idx]
            # train_val_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)

            source, target = splits['valid']['source_node'],  splits['valid']['target_node']
            pos_valid_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
            neg_valid_edge = splits['valid']['target_node_neg'] 

            source, target = splits['test']['source_node'],  splits['test']['target_node']
            pos_test_edge = torch.cat([source.unsqueeze(0), target.unsqueeze(0)], dim=0)
            neg_test_edge = splits['test']['target_node_neg']

        loggers = {
            'Hits@20': Logger(cfg.train.runs),
            'Hits@50': Logger(cfg.train.runs),
            'Hits@100': Logger(cfg.train.runs),
            'MRR': Logger(cfg.train.runs),
            'AUC':Logger(cfg.train.runs),
            'AP':Logger(cfg.train.runs),
            'mrr_hit20':  Logger(cfg.train.runs),
            'mrr_hit50':  Logger(cfg.train.runs),
            'mrr_hit100':  Logger(cfg.train.runs),
        }
                
        idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
        train_val_edge = pos_train_edge[idx]

        evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]
        
        for run in range(cfg.train.runs):

            print('#################################          ', run, '          #################################')
            if cfg.train.runs == 1:
                seed = args.seed
            else:
                seed = run
            print('seed: ', seed)

            seed_everything(seed)
            
            save_path = cfg.save.output_dir+'/lr'+str(cfg.train.lr) \
                + '_drop' + str(cfg.model.dropout) + '_l2'+ \
                    str(cfg.train.l2) + '_numlayer' + str(cfg.model.num_layers)+ \
                        '_numPredlay' + str(cfg.score_model.num_layers_predictor) +\
                            '_numGinMlplayer' + str(cfg.score_model.gin_mlp_layer)+ \
                                '_dim'+str(cfg.model.hidden_channels) + '_'+ 'best_run_'+str(seed)

            if emb != None:
                torch.nn.init.xavier_uniform_(emb.weight)

            model.reset_parameters()
            score_func.reset_parameters()

            if emb != None:
                optimizer = torch.optim.Adam(
                    list(model.parameters()) + list(score_func.parameters()) + list(emb.parameters() ),lr=cfg.train.lr, weight_decay=cfg.train.l2)
            else:
                optimizer = torch.optim.Adam(
                        list(model.parameters()) + list(score_func.parameters()),lr=cfg.train.lr, weight_decay=cfg.train.l2)
            best_valid = 0
            kill_cnt = 0
            
            for epoch in range(1, 1 + cfg.train.epochs):
                loss = train(model, 
                            score_func, 
                            pos_train_edge, 
                            dataset._data, 
                            emb, 
                            optimizer, 
                            cfg.train.batch_size, 
                            train_edge_weight, 
                            device)

                # for attention score   
                # print(model.convs[0].att_src[0][0][:10])
                
                if epoch % 100 == 0:
                    results_rank, score_emb = test(model, 
                                                score_func,
                                                splits['test'],
                                                evaluation_edges, 
                                                emb, 
                                                evaluator_hit, 
                                                evaluator_mrr, 
                                                cfg.train.batch_size, 
                                                cfg.data.name, 
                                                cfg.train.use_valedges_as_input, 
                                                device)


                    for key, _ in loggers.items():
                        loggers[key].add_result(run, results_rank[key])

                    if epoch % 100 == 0:
                        for key, result in results_rank.items():
                            train_hits, valid_hits, test_hits = result
                            
                    logging.info(
                        f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')
                    
                    r = torch.tensor(loggers[eval_metric].results[run])
                    best_valid_current = round(r[:, 1].max().item(),4)
                    best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                    print(eval_metric)
                    
                    logging.info(f'best valid: {100*best_valid_current:.2f}%, '
                                    f'best test: {100*best_test:.2f}%')
                    
                    if len(loggers['AUC'].results[run]) > 0:
                        r = torch.tensor(loggers['AUC'].results[run])
                        best_valid_auc = round(r[:, 1].max().item(), 4)
                        best_test_auc = round(r[r[:, 1].argmax(), 2].item(), 4)
                        
                        print('AUC')
                        logging.info(f'best valid: {100*best_valid_auc:.2f}%, '
                                    f'best test: {100*best_test_auc:.2f}%')
                    
                    print('---')

                    if best_valid_current > best_valid:
                        best_valid = best_valid_current
                        kill_cnt = 0
                        if cfg.save: save_emb(score_emb, save_path)
                    else:
                        kill_cnt += 1
                        
                        if kill_cnt > cfg.train.kill_cnt: 
                            print("Early Stopping!!")
                            break
            
            for key in loggers.keys():
                if len(loggers[key].results[0]) > 0:
                    print(key)
                    loggers[key].print_statistics( run)
                    print('\n')
            
      
    
        result_all_run = {}
        for key in loggers.keys():
            if len(loggers[key].results[0]) > 0:
                print(key)
                best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()
                if key == eval_metric:
                    best_metric_valid_str = best_metric
                    # best_valid_mean_metric = best_valid_mean
                if key == 'AUC':
                    best_auc_valid_str = best_metric
                    # best_auc_metric = best_valid_mean
                result_all_run[key] = [mean_list, var_list]
                

            
        if cfg.train.runs == 1:
            print(str(best_valid_current) + ' ' + str(best_test) + ' ' + str(best_valid_auc) + ' ' + str(best_test_auc))
        
        else:
            print(str(best_metric_valid_str) + ' ' +str(best_auc_valid_str))
