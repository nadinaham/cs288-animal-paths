#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p gpu,davies_gpu,gpu_requeue
#SBATCH -t 3-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128G          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH --gres gpu:1
#SBATCH -o bash-outputs/experiment4-davies_sep_mask_800_fuse_thermal-rgb_100-0/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env/

module load gcc/13.2.0-fasrc01

## arguments for python script, see python script called for thier specifications
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-400px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-1200px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-400px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/rgb-tiles/png-images"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-1200px/rgb-tiles/png-images"
## Experiment 4: davies
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-10-90"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-20-80"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-30-70"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-40-60"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-50-50"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-60-40"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-70-30"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-80-20"
# img_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/fused/tr-fused-90-10"
image_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/thermal-tiles/png-images"

# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/path-masks-finished/masks_tog_fixed"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/path-masks-finished/masks_tog_dilated"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-400px/path-masks-finished"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/path-masks-finished"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-1200px/path-masks-finished"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-400px/path-masks-finished/masks_sep"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-800px/path-masks-finished/masks_sep"
# mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/clipped-relabeled-tiles/tiles-1200px/path-masks-finished/masks_sep"
## Experiment 4: davies
mask_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-800px/path-masks-finished"

# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-5/SYN_tog_800px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-5/davies_tog_800px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-1/davies_sep_400px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-1/davies_sep_800px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-1/davies_sep_1200px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-2/SYN_sep_400px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-2/SYN_sep_800px_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-2/SYN_sep_1200px_output_segmtask"
## Experiment 4: davies
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_10-90_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_20-80_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_30-70_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_40-60_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_50-50_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_60-40_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_70-30_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_80-20_output_segmtask"
# output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_90-10_output_segmtask"
output_dir="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/experiment_outputs/experiment-4/davies_sep_800px_fused_tr_100-0_output_segmtask"

# commented out because does not work on Sam's account
# set -x
# date
# source ~/.bashrc
python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/models/detectron2/train_test_custom_data.py $img_dir $mask_dir $output_dir

## To run this: make sure your directory is the parent directory of this project
## (in the line below, it is the parent directory of the outer AnimalPathMapping folder)
## then enter in the terminal: 'sbatch AnimalPathMapping/AnimalPathMapping/models/detectron2/run_train_test_custom_data.sh'