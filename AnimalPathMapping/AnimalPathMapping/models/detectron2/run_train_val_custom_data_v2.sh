#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p gpu,davies_gpu,gpu_requeue
#SBATCH -t 3-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128G          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH --gres gpu:1
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

## Commented out to work on Sam's account
# source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
# conda deactivate
# conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env/

# module load gcc/13.2.0-fasrc01

## Added (replaced lines commented out above) to work on Sam's account
source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env/

module load gcc/13.2.0-fasrc01

## arguments for python script, see python script called for their specifications
img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/rgb-tiles/png-images"
mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/path-masks-finished"
output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/train_val_outputs/train_val_800px_davies_lbls_output"

## Commented out to work on Sam's account
# set -x
# date
# source ~/.bashrc
## python ./train_val_custom_data_v2.py $img_dir $mask_dir $output_dir
python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/models/detectron2/train_val_custom_data_v2.py $img_dir $mask_dir $output_dir

## To run this, make sure your current working directory is the outermost directory of this project.
## Enter 'sbatch run_train_val_custom_data_v2.sh' in the terminal