#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --partition=cpuonly
#SBATCH --job-name=tag_struc2vec
#SBATCH --mem-per-cpu=1600mb

#SBATCH --output=log/TAG_Benchmark_%j.output
#SBATCH --error=error/TAG_Benchmark_%j.error


#SBATCH --chdir=/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE/scripts

# Notification settings:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cc7738@kit.edu
source /hkfs/home/project/hk-project-test-p0021478/cc7738/anaconda3/etc/profile.d/conda.sh

conda activate base
conda activate nui
cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns
# <<< conda initialize <<<
module purge
module load devel/cmake/3.18
module load devel/cuda/11.8
module load compiler/gnu/12


# python wb_tune.py --cfg core/yamls/cora/gat.yaml --sweep core/yamls/cora/gat_sp1.yaml
# python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml
# python wb_tune.py --cfg core/yamls/pubmed/gcns/gae.yaml --sweep core/yamls/pubmed/gcns/gae_sp1.yaml


# python wb_tune.py --cfg core/yamls/arxiv_2023/gcns/gae.yaml --sweep core/yamls/arxiv_2023/gcns/gae_sp1.yaml

# python wb_tune.py --cfg core/yamls/ogbn-arxiv/gcns/gae.yaml --sweep core/yamls/ogbn-arxiv/gcns/gae_sp1.yaml

# python wb_tune.py --cfg core/yamls/ogbn-products/gcns/gae.yaml --sweep  core/yamls/ogbn-products/gcns/gae_sp1.yaml

cd /hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns
CUDA_VISIBLE_DEVICES=0 python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml 
CUDA_VISIBLE_DEVICES=1 python wb_tune.py --cfg core/yamls/pubmed/gcns/gae.yaml --sweep core/yamls/pubmed/gcns/gae_sp1.yaml
CUDA_VISIBLE_DEVICES=2 python wb_tune.py --cfg core/yamls/arxiv_2023/gcns/gae.yaml --sweep core/yamls/arxiv_2023/gcns/gae_sp1.yaml 
CUDA_VISIBLE_DEVICES=2 python wb_tune.py --cfg core/yamls/ogbn-arxiv/gcns/gae.yaml --sweep core/yamls/ogbn-arxiv/gcns/gae_sp1.yaml
CUDA_VISIBLE_DEVICES=3 python wb_tune.py --cfg core/yamls/ogbn-products/gcns/gae.yaml --sweep  core/yamls/ogbn-products/gcns/gae_sp1.yaml 


python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml --data 'cora'> cora-output.txt
python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml --data 'pubmed'> pubmed-output.txt
# python wb_tune.py --cfg core/yamls/ogbn-arxiv/gcns/gae.yaml --sweep core/yamls/ogbn-arxiv/gcns/gae_sp1.yaml --data 'ogbn-arxiv' > ogbn-arxiv-output.txt 
# problem 
python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml --data 'ogbn-arxiv' > ogbn-arxiv2-output.txt
python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml --data 'ogbn-products' > ogbn-products-output.txt
python wb_tune.py --cfg core/yamls/cora/gcns/gae.yaml --sweep core/yamls/cora/gcns/gae_sp1.yaml --data 'arxiv_2023' > arxiv_2023-output.txt

# python wb_tune.py --cfg core/yamls/ogbn-products/gcns/gae.yaml --sweep  core/yamls/ogbn-products/gcns/gae_sp1.yaml --device 2 > ogbn-product-output2.txt 
# python wb_tune.py --cfg core/yamls/pubmed/gcns/gae.yaml --sweep core/yamls/pubmed/gcns/gae_sp1.yaml --device 3 > pubmed-output2.txt 