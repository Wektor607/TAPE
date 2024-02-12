import os
import numpy as np
import time
import datetime
import pytz


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path):
        return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# * ============================= Time Related =============================


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper

def get_root_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(file_dir, "..")

import git
import subprocess

def get_git_repo_root_path():
    try:
        # Using git module
        git_repo = git.Repo('.', search_parent_directories=True)
        return git_repo.working_dir
    except git.InvalidGitRepositoryError:
        # Fallback to using subprocess if not a valid Git repository
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print("Error:", result.stderr)
            return None

import pandas as pd
import torch 
import csv
import uuid

# Define a function that uses the lambda function
def process_value(v):
    return (lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)(v)

from IPython import embed
def append_acc_to_excel(metrics_acc, root, name):
    # if not exists save the first row
    
    csv_columns = ['Metric'] + list(k for k in metrics_acc) 

    try:
        Data = pd.read_csv(root)[:1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)
    
    # transform dict to df    
    id = uuid.uuid4()
    acc_lst = []
    
    for k, v in metrics_acc.items():
        acc_lst.append(process_value(v))
        
    
    v_lst = [f'{name}_{id}'] + acc_lst
    new_df = pd.DataFrame([v_lst], columns=csv_columns)
    Data = pd.concat([Data, new_df])
    highest_values = Data.apply(lambda column: max(column, default=None))

    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=Data.columns)
    Data = pd.concat([Data, Best_df])
    Data.to_csv(root,index=False)

    return Data


def append_mrr_to_excel(metrics_mrr, root):
    # if not exists save the first row
    # transform dict to df    
    csv_columns, csv_numbers = [], []
    for i, (k, v) in enumerate(metrics_mrr.items()): 
        if i == 0:
            csv_columns = ['Metric'] + list(v.keys())
        csv_numbers.append([k] + list(v.values()))
    
    print(csv_numbers)

    try:
        Data = pd.read_csv(root)[:1]
    except:
        Data = pd.DataFrame(None, columns=csv_columns)
        Data.to_csv(root, index=False)

    
    new_df = pd.DataFrame(csv_numbers, columns = csv_columns)
    Data = pd.concat([Data, new_df])
    
    highest_values = Data.apply(lambda column: max(column, default=None))
    Best_list = ['Best'] + highest_values[1:].tolist()
    Best_df = pd.DataFrame([Best_list], columns=csv_columns)
    Data = pd.concat([Data, Best_df])
    
    Data.to_csv(root, index=False)

    
    return Data