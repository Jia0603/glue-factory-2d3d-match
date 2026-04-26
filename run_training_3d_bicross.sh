#!/bin/bash
#SBATCH -A berzelius-2026-113 
#SBATCH -J nll.2      
#SBATCH -t 01-00:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/3D_aug_nll_0.2_bat32_1e-5%j.log

#SBATCH -p berzelius
#SBATCH --nodes=1
#SBATCH --nodelist=node[061-063,065-082,084-093] 
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train_new --mp bfloat16 \
    --conf gluefactory/configs/2d_3d_lightglu3D_bicross_SP_finetune_n.yaml 3D_aug_nll_0.2_bat32_1e-5 \
    --distributed --no_eval_0 --restore