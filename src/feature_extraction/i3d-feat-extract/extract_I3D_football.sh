#!/bin/bash
#SBATCH --job-name Foot_I3D
#SBATCH --array=0-499%12
#SBATCH --time=4:00:00 
#SBATCH --mem=60G
#SBATCH --gres=gpu:1 --constraint="[titan_x_p|titan_x_m]"
#SBATCH -o logs/output.%3a.%A.out
#SBATCH -e logs/output.%3a.%A.err

module purge
module load applications-extra
module load cuda/8.0.61-cudNN5.1
module load anaconda3

echo "Activating i3d-feat-extract ..."
source activate i3d-feat-extract
echo "i3d-feat-extract activated."

echo "Exporting PYTHONPATH ..."
export PYTHONPATH='/home/giancos/i3d-feat-extract/kinetics-i3d':$PYTHONPATH
echo "PYTHONPATH exported."

echo "Defining args..."
video_dir='/vcc/datasets/football/dataset_crop224/'
feat_dir='/vcc/datasets/football/dataset_crop224/'

cd /home/giancos/i3d-feat-extract
echo "args defined."

python extract_i3d_spatial_features.py $video_dir $feat_dir --jobid $SLURM_ARRAY_TASK_ID
