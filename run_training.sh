#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J train_fr_8bat_10ep_lg3d               
#SBATCH -t 00-10:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/train_lightglu3d%j.log

#SBATCH -p berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train --conf gluefactory/configs/2d_3d_lightglue_SP_finetune.yaml lightglu3d_fr_8bat_10ep_finetune