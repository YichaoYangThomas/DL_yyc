#SBATCH --job-name=job_wgpu
#SBATCH --account=ds_ga_1008_002-2025sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --requeue

singularity exec --bind /scratch --nv --overlay /scratch/yy5020/overlay-25GB-500K.ext3:rw /scratch/yy5020/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /scratch/yy5020/nlp_tutorial/
python ./train.py
