#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J lightglu3d       
#SBATCH -t 00-10:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/lightglu3d_distribute_4thread_72sces_flash_bat32_1e-3%j.log

#SBATCH -p berzelius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train --mp bfloat16 \
    --conf gluefactory/configs/2d_3d_lightglu3D_SP_finetune.yaml new_lightglu3d_4thread_72sces_flash_bat32_1e-3 \
    --distributed --no_eval_0