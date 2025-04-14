#!/bin/bash
#SBATCH --job-name=jepa_train
#SBATCH --account=ds_ga_1008_002-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

# 打印作业信息
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURMD_NODENAME"
echo "分配的GPU: $CUDA_VISIBLE_DEVICES"
echo "开始时间: $(date)"

# 进入Singularity环境并运行训练
singularity exec --bind /scratch --nv --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw /scratch/$USER/ubuntu-20.04.3.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
conda activate my_env

# 进入项目目录
cd /scratch/$USER/DL_Final_Proj-main/

# 运行训练脚本
echo '开始训练...'
python ./train.py

# 输出模型信息
echo '检查模型保存情况:'
ls -lh ./model_weights.pth

# 输出GPU使用情况
nvidia-smi
"

echo "结束时间: $(date)" 