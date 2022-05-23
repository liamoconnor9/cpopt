#PBS -S /bin/bash
#PBS -l select=2:ncpus=40:mpiprocs=40:model=sky_ele
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -W group_list=s2276
file=${0##*/}
job_name="${file%.*}"
#PBS -N ${job_name}

export PATH=$HOME/scripts:$PATH
deactivate
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda activate dedalus-d3
# support lots of text output to stdio for analysis
export MPI_UNBUFFERED_STDIO=true

# source ~/png2mp4.sh
cd ~/vpopt

FILE="$(readlink -f "$0")"
DIR="$(dirname "$(readlink -f "$0")")/"
CONFIG="nsvp_options.cfg"

mpirun -np 16 python3 ns_airfoil.py