#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J lightglu3d       
#SBATCH -t 00-20:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/lightglu3d_bicross_1thread_flash_bat16_1e-5_2Ddecay_50_100%j.log

#SBATCH -p berzelius
#SBATCH --nodes=1
#SBATCH --nodelist=node[061-063,065-082,084-093] 
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train_new --mp bfloat16 \
    --conf gluefactory/configs/2d_3d_lightglu3D_bicross_SP_finetune.yaml lightglu3d_bicross_1thread_flash_bat16_1e-5_50_100 \
    --distributed --no_eval_0 --restore