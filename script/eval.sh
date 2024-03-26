#!/bin/bash

# this script is used for training MonoHuman on euler

#SBATCH -n 8
#SBATCH -t 00-04
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus=v100:1
#SBATCH --gres=gpumem:32G
#SBATCH --tmp=8G
#SBATCH -A es_tang
#SBATCH --output=logs/slurm-%j.out
#SBATCH --mail-type=FAIL

SEQ=$1

cd ..
#export WANDB_API_KEY=$(cat "wandb.key")
module load gcc/9.3.0 cuda/10.1.243 eth_proxy
python run.py --type eval --cfg configs/monohuman/zju_mocap/${SEQ}/${SEQ}.yaml