#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J 47scene       
#SBATCH -t 00-18:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/newest_2thread_bat32_1e-5_47scenes%j.log

#SBATCH -p berzelius
#SBATCH --nodes=1
#SBATCH --nodelist=node[061-063,065-082,084-093] 
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train_new --mp bfloat16 \
    --conf gluefactory/configs/2d_3d_lightglu3D_bicross_SP_finetune.yaml newest_2thread_bat32_1e-5_47scenes \
    --distributed --no_eval_0 --restore