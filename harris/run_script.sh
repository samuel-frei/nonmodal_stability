#!/bin/bash

#SBATCH --job-name=mug
#SBATCH --nodes=1
#SBATCH -t 0-02:00

source /jet/home/freiberg/.bashrc
cd $PROJECT/pseudospectra/harris/

OMP_NUM_THREADS=128 /jet/home/freiberg/OpenFUSIONToolkit/build_release/examples/MUG/harris_sheet/harris_sheet oft_surf.in oft_in.xml
