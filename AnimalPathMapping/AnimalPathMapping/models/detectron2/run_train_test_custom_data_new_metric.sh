#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p gpu,davies_gpu,gpu_requeue
#SBATCH -t 3-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128G          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH --gres gpu:1
## #SBATCH -o bash-outputs/experiment5-davies_tog_mask_800_new-metrics/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -o bash-outputs/experiment4-SYN_sep_mask_800_tr_20-80_new-metrics_w-images/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env/

module load gcc/13.2.0-fasrc01

## arguments for python script, see README.md and python script called for their specifications

img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/fused/tr-fused-20-80"

mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/path-masks-finished/masks_sep"

output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/SYN_sep_800px_fused_tr_20-80_output_new-metrics_w-images"


# commented out because does not work on Sam's account
# set -x
# date
# source ~/.bashrc
python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/models/detectron2/train_test_custom_data_new-metric.py $img_dir $mask_dir $output_dir

## To run this: make sure your directory is the parent directory of this project
## (in the line below, it is the parent directory of the outer AnimalPathMapping folder)
## then enter in the terminal: 'sbatch AnimalPathMapping/AnimalPathMapping/models/detectron2/run_train_test_custom_data.sh'