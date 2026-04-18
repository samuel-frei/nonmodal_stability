#!/usr/bin/env bash
#SBATCH --job-name=ps1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=0-08:00
#SBATCH --output=/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear_20x6z/output_default.log
#SBATCH --error=/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear_20x6z/error_default.log
#SBATCH --mail-user=swf2112@columbia.edu
#SBATCH --mail-type=ALL

source /jet/home/freiberg/.bashrc
module load anaconda3
conda activate openfusion

cd /ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear_20x6z

set -euo pipefail

echo "Starting case=default at \Sun Apr  5 15:48:04 EDT 2026"
srun python /ocean/projects/phy240045p/freiberg/pseudospectra/scripts/lin_ops_full_reduction.py \
  --grid-points 1024 \
  --nprocs 128 \
  --min-level 1e-12 \
  --nlevels 8 \
  --run-tag default \
  --case-tag default \
  --real-min ' -1.E7' --real-max ' 0.1' --imag-min ' -3.E6' --imag-max ' 3.E6' \
  --output-dir /ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear_20x6z \
  --plot-name pseudoplot_default.png
echo "Finished case=default at \Sun Apr  5 15:48:04 EDT 2026"
