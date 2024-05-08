#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p davies
#SBATCH -t 0-08:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128GB          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## module load Anaconda3/2020.11

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/AnimalPathMapping/

module load gcc/13.2.0-fasrc01

## arguments for python script, see python script called for thier specifications
thermal_tiff_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/thermal.tif" 
rgb_tiff_path="None"
lidar_tiff_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/lidar.tif"
mask_tiff_path="None"
save_thermal="None"
save_rgb="None"
save_lidar="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/just-lidar-orthomosaic.npy"
save_mask="None"



python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/align_orthomosaics.py $thermal_tiff_path $rgb_tiff_path $lidar_tiff_path $mask_tiff_path $save_thermal $save_rgb $save_lidar $save_mask