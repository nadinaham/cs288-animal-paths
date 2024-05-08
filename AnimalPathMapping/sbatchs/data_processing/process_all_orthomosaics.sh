#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p sapphire,davies
#SBATCH -t 0-08:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128GB          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## module load Anaconda3/2020.11

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/AnimalPathMapping/

module load gcc/13.2.0-fasrc01

## arguments for python script, see python script called for their specifications
thermal_tiff_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/thermal.tif" 
rgb_tiff_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/rgb_original.tif"
lidar_tiff_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/lidar.tif"
mask_tiff_path="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/images/rasterized_paths_by_corrected_id_grayscale.tif"
save_thermal="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/thermal-orthomosaic.npy"
save_rgb="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/rgb-orthomosaic.npy"
save_lidar="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/lidar-orthomosaic.npy"
save_mask="/n/davies_lab/Lab/shared_projects/AnimalPathMapping/firestorm-3/orthomosaics/path-mask-orthomosaic.npy"



python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/align_orthomosaics.py $thermal_tiff_path $rgb_tiff_path $lidar_tiff_path $mask_tiff_path $save_thermal $save_rgb $save_lidar $save_mask