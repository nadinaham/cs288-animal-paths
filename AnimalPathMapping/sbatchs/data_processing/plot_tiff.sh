#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p davies
#SBATCH -t 0-02:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 32GB          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o bash-outputs/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## module load Anaconda3/2020.11

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/AnimalPathMapping/

module load gcc/13.2.0-fasrc01

## Needed to get rid of these three lines below for it to work
## set -x
## date
## source ~/.bashrc
python /n/davies_lab/Lab/shared_projects/AnimalPathMapping/cs288-animal-paths/AnimalPathMapping/AnimalPathMapping/data_processing/process_tiff.py /n/davies_lab/Lab/shared_projects/AnimalPathMapping/images/rasterized_paths_by_corrected_id_grayscale.tif /n/davies_lab/Lab/shared_projects/AnimalPathMapping/images/path_by_id.png