#!/usr/bin/env bash
#SBATCH --job-name=ps1
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00
#SBATCH --output=/mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z/pseudo_marginal/output_default.log
#SBATCH --error=/mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z/pseudo_marginal/error_default.log

cd /mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z/pseudo_marginal

set -euo pipefail

srun python /mnt/homes/sfreiberger/research/recon_data/psc_data/scripts/lin_ops_full_reduction.py \
  --grid-points 3200 \
  --nprocs 16 \
  --adaptive-grid \
  --min-level 1e-7 \
  --run-tag default \
  --case-tag default \
  --real-min ' -9.E6' --real-max ' 5.e5' --imag-min ' -3.E6' --imag-max ' 3.E6' \
  --output-dir /mnt/homes/sfreiberger/research/recon_data/psc_data/harris_linear_20x6z/pseudo_marginal

