#!/bin/bash
#SBATCH -J TensorFlow
#SBATCH --partition=defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=20

# If modules are needed by the script, then source modules environment:
. /etc/profile.d/modules.sh
. $HOME/venv366tfcpu/bin/activate
#module add cuda80/toolkit/8.0.61
# Work directory
#workdir="$SLURM_SUBMIT_DIR"
workdir="/home/david/git-repos/brain"

# Full path to application + application name
application="$(which python)"

# Run options for the application
options="$workdir/main.py"

###############################################################
### You should not have to change anything below this line ####
###############################################################
# change the working directory (default is home directory)
cd $workdir
echo Running on host $(hostname)
echo Time is $(date)
echo Directory is $(pwd)
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo $SLURM_JOB_NODELIST

command="$application $options"

# Run the executable
ulimit -s unlimited
export OMP_NUM_THREADS=20
echo Running $command
time $command
