#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J lightglu3d_adapt       
#SBATCH -t 00-24:00:00               
#SBATCH -o /home/x_lishu/matching/colla_gluefactory/glue-factory-2d3d-match/outputs/log_file/lightglu3d_2thread_flash_bat32_1e-4_2Ddecay%j.log

#SBATCH -p berzelius
#SBATCH --nodes=1
#SBATCH --nodelist=node[061-063,065-082,084-093] 
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train lightglue_adapt_v2     --conf gluefactory/configs/2d_3d_lightglue_SP_finetune.yaml