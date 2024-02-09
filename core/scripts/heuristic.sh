cd /hkfs/work/workspace_haic/scratch/cc7738-TAG/core
module load devel/cuda/11.8

source /hkfs/home/haicore/aifb/cc7738/anaconda3/etc/profile.d/conda.sh
conda activate TAPE
module unload jupyter/tensorflow/2023-09-12
module load devel/cuda/11.4
# module load devel/cuda/11.4
module load devel/cmake/3.18
module load compiler/intel/2021.4.0
module unload numlib/mkl/2020
module unload mpi/openmpi/4.1



python cora_heuristic.py 
python arxiv_heuristic.py
python ogbn_products_heuristic.py
python pubmed_heuristic.py
