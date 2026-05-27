#!/bin/bash
#SBATCH -A berzelius-2026-113 
#SBATCH -J diff_train       
#SBATCH -t 00-01:00:00               
#SBATCH -o /home/x_jiagu/degree_project/log_file/test.log

#SBATCH -p berzelius
#SBATCH --nodes=1
#SBATCH --nodelist=node[061-063,065-082,084-093] 
#SBATCH --ntasks-per-node=1                 
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=384G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python -m gluefactory.train_new \
    --conf gluefactory/configs/2d_3d_lightglu3D_bicross_SP_finetune_n.yaml \
   test \
    --mp bfloat16 \
    --distributed \
    --no_eval_0