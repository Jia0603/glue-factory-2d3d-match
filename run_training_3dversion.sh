#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J lightglu3d 
        
#SBATCH -t 00-24:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/lightglu3d_test%j.log

#SBATCH -p berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train --conf gluefactory/configs/2d_3d_lightglu3D_SP_finetune.yaml new_lightglu3d_test