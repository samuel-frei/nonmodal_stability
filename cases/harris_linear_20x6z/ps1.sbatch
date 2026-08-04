#!/usr/bin/env bash
#SBATCH --job-name=ps1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00
#SBATCH --output=output_%x_%j.log
#SBATCH --error=error_%x_%j.log

# Submit from the directory holding lin_ops.h5 and mass_mat.h5, e.g.
#
#   cd /path/to/harris_linear_20x6z/pseudo_marginal
#   sbatch /path/to/nonmodal/cases/harris_linear_20x6z/ps1.sbatch
#
# Everything is resolved relative to the submission directory, so this file
# carries no absolute paths and works for any user on any cluster.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# `nonmodal` is the console script installed by this package. Activate the
# environment that provides it before submitting, or point RUN at `uv run`:
#   RUN="uv run --project /path/to/nonmodal nonmodal"
RUN="${RUN:-nonmodal}"

srun ${RUN} \
  --grid-points 3200 \
  --nprocs "${SLURM_CPUS_PER_TASK:-16}" \
  --min-level 1e-7 \
  --nlevels 16 \
  --run-tag "${SLURM_JOB_NAME:-default}" \
  --case-tag harris_linear_20x6z \
  --real-min -9.0e6 --real-max 5.0e5 \
  --imag-min -3.0e6 --imag-max 3.0e6 \
  --jacobian ./lin_ops.h5 \
  --massmat ./mass_mat.h5 \
  --cache-dir . \
  --output-dir .
