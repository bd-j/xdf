#!/bin/bash

#SBATCH -J forcepho_xdf
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t  1:00:00 # Runtime
#SBATCH -p itc,shared # Partition to submit to
#SBATCH --constraint="intel"
#SBATCH --mem-per-cpu=500 #in MB 
#SBATCH -o /n/regal/eisenstein_lab/bdjohnson/xdf/logs/xdf_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/regal/eisenstein_lab/bdjohnson/xdf/logs/xdf_%A_%a.err # Standard err goes to this file

source activate force

xdf=/n/regal/eisenstein_lab/bdjohnson/xdf

regfile=${xdf}/scripts/xdf_regions.dat 
backend=pymc3
nwarm=500
niter=250

reg=${SLURM_ARRAY_TASK_ID}

out=results/xdf_reg${reg}


unset regions
while IFS= read -r line; do
    arr=($line)
    corners=${arr[@]:0:4}
    regions+=("$corners")
done < $regfile
corner=${regions[$reg+1]}


cd ${xdf}/scripts
echo xdf_multi_force.py --xdf_dir=$xdf --results_name=$out\
       --backend=$backend --nwarm=$nwarm --niter=$niter \
       --corners ${corner} --filters f814w f160w



