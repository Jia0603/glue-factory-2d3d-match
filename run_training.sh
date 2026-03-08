#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J train_lightglu3d 
#SBATCH --gres=gpu:a100-sxm4-80gb:1 
#SBATCH -w node[061-093]         
#SBATCH -t 00-24:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/train_lightglu3d_unfre%j.log

#SBATCH -p berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train --conf gluefactory/configs/2d_3d_lightglue_SP_finetune.yaml lightglu3d_train_all_18scns_16bat_10ep_1e-3 --restore 