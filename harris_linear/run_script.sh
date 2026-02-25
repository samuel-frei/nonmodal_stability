#!/bin/bash

#SBATCH --job-name=mug
#SBATCH --nodes=1
#SBATCH -t 0-00:15

source /jet/home/freiberg/.bashrc
cd $PROJECT/pseudospectra/harris_linear/

OMP_NUM_THREADS=128 /jet/home/freiberg/OpenFUSIONToolkit/build_release/examples/MUG/harris_sheet/harris_sheet_linear oft_surf.in oft_in.xml
