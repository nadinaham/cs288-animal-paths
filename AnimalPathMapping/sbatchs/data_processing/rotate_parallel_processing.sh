#!/bin/bash -x

#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p davies
#SBATCH -t 0-00:80:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem 128GB          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o bash-outputs/rotate-parallel/%A-%a.out  # File to which STDOUT will be written, %A inserts jobid %a inserts array id
#SBATCH -e bash-errors/rotate-parallel/%A-%a.err  # File to which STDERR will be written, %A inserts jobid %a inserts array id

## NOTE: change this to index which jobs from a job array file you would like to run,
## should be 1-<line number of last line in job array file>
## the job array file for this is 'rotation_parallel_processing_cmds.sh', 
## the output of running 'bash get_rotation_parallel_processing_cmds.sh'
#SBATCH --array 1-2705

source /n/sw/eb/apps/centos7/Anaconda3/2020.11/etc/profile.d/conda.sh
## module load Anaconda3/2020.11

conda deactivate
conda activate /n/davies_lab/Lab/shared_conda_envs/detectron_env

module load gcc/13.2.0-fasrc01

## Run the job array (submits all the jobs in parallel to the cluster)
awk 'NR=='"$SLURM_ARRAY_TASK_ID"'' rotation_parallel_processing_cmds_both.sh | bash