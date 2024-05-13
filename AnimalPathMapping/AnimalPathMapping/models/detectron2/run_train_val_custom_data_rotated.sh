#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p gpu,davies_gpu,gpu_requeue
#SBATCH -t 2-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128G          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH --gres gpu:1
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## module load Anaconda3/2020.11

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env/

module load gcc/13.2.0-fasrc01

## source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## conda deactivate
# conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env/

## module load gcc/13.2.0-fasrc01

## arguments for python script, see python script called for their specifications
img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/rotated/rgb-tiles/png-images"
mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/rotated/path-masks-finished/masks_sep"
output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/train_val_outputs/train_val_800px_rotations_manual_lbls_output"

## set -x
## date
## source ~/.bashrc
## python ./train_val_custom_data_v2.py $img_dir $mask_dir $output_dir
python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/models/detectron2/train_val_test_custom_data_rotated.py $img_dir $mask_dir $output_dir