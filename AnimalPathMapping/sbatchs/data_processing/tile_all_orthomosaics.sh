#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p sapphire,davies
#SBATCH -t 3-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 150GB          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## module load Anaconda3/2020.11

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env

module load gcc/13.2.0-fasrc01

## arguments for python script, see python script called for thier specifications
## thermal_orthomosaic_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/thermal-orthomosaic-matrix.npy" # Lucia's
thermal_orthomosaic_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/thermal-orthomosaic.npy" # from Sam's processing
## rgb_orthomosaic_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/rgb-orthomosaic-matrix.npy" # Lucia's
rgb_orthomosaic_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/rgb-orthomosaic.npy"  # from Sam's processing
lidar_orthomosaic_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/lidar-orthomosaic.npy" # from Sam's processing (Lucia's corrupted)
midden_orthomosaic_path="None"
path_mask_orthomosaic_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/mask-orthomosaic.npy"
thermal_processing_constants_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/thermal-orthomosaic-thermal_processing_info.npy"
output_folder="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/image-tiles/all-tiles-1200px"
save_thermal="True"
save_rgb="True"
save_lidar="True"
save_midden_mask="False"
save_path_mask="True"

python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/tile_orthomosaic.py $thermal_orthomosaic_path $rgb_orthomosaic_path $lidar_orthomosaic_path $midden_orthomosaic_path $path_mask_orthomosaic_path $thermal_processing_constants_path $output_folder $save_thermal $save_rgb $save_lidar $save_midden_mask $save_path_mask