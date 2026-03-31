#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J lightglu3d_adapt       
#SBATCH -t 00-06:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/lightglue+adapter_2thread_bat32_1e-5_47scenes%j.log

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
    --conf gluefactory/configs/2d_3d_lightglue_adapt_SP_finetune.yaml lightglue+adapter_2thread_bat32_1e-5_47scenes \
    --distributed --no_eval_0