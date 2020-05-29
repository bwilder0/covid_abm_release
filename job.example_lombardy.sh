#!/bin/bash -x
#SBATCH -J lombardy_bayesian # A single job name for the array
#SBATCH -n 1                # Number of tasks
#SBATCH --cpus-per-task 4  # How to allocate CPUs when multithreading
#SBATCH --threads-per-core 1 # minimum number of threads in a core to specify to a job
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -p shared
#SBATCH -t 5:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH --mem=10000          # Memory pool for all cores (see also --mem-per-cpu) MBs
#SBATCH -o ../joblogs/%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ../joblogs/%A_%a.err  # File to which STDERR will be written, %j inserts jobid

THREADS=4 # should be set to the same as -c above

set -x

date
cdir=$(pwd)
#tempdir="/scratch/bwilder/"
#mkdir -p $tempdir
#cd $tempdir

echo ${SLURM_ARRAY_TASK_ID}

python3 parameter_sweep_lombardy.py --N_SIMS_PER_COMBO $1 --N_SIMS_PER_JOB $2 --index ${SLURM_ARRAY_TASK_ID} --sim_name $3 --seed_offset $4 --EXP_ROOT_DIR /n/holylfs/LABS/tambe_lab/bwilder/covid19/parameter_sweeps/ -t $THREADS --popdir /n/holylfs/LABS/tambe_lab/covid_populations/ -N 10000000
