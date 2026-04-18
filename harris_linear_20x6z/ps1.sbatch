#!/usr/bin/env bash
#SBATCH --job-name=ps1
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00
#SBATCH --output=/mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z/output_default.log
#SBATCH --error=/mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z/error_default.log

cd /mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z

set -euo pipefail

srun python /mnt/homes/sfreiberger/research/recon_data/psc_data/scripts/lin_ops_full_reduction.py \
  --grid-points 64 \
  --nprocs 32 \
  --min-level 1e-12 \
  --nlevels 8 \
  --run-tag default \
  --case-tag default \
  --real-min ' -1.E7' --real-max ' 1.e6' --imag-min ' -3.E6' --imag-max ' 3.E6' \
  --output-dir /mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z \
  --plot-name pseudoplot_default.png
