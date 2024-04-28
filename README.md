# Benchmark TAG 

<img src="./overview.svg">


## 0.0 Python environment setup with Conda
```
conda create --name TAPE python=3.8
conda activate TAPE

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
pip install --upgrade accelerate
```

## 0.1 Here is my install examples in horeka server
```

Currently Loaded Modules:
  1) devel/cmake/3.18   2) devel/cuda/10.2   3) devel/cudnn/10.2 (E)   4) compiler/gnu/11.1

  Where:
   E:  Experimental

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu114
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c dglteam/label/cu113 dgl
pip install yacs
pip install transformers
pip install --upgrade accelerate

```

## 1. Download/Test TAG datasets 

```
bash core/scripts/get-tapedataset.sh 
python load_arxiv_2023.py 
python load_cora.py
python load_ogbn_arxiv.py
python load_products.py
python load_pubmed.py
#TODO add paperwithcode dataset
#TODO use SemOpenAlex
```

### A. Original Text Attributes
All graph encoder modules including node encoder and edge encoder are implemented in GraphGym transferred from [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/graphgym.html#).

#### FeatNodeEncoder

### B. LLM responses

## 2. Fine-tuning the LMs
### To use the orginal text attributes
### To use the GPT responses



## 3. Training the GNNs
### To use different GNN models

## 4. Reproducibility
