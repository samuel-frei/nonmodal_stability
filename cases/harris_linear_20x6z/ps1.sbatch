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

srun ${RUN} run \
  --grid-nx 24 --grid-ny 24 \
  --refine-points 2600 \
  --nprocs "${SLURM_CPUS_PER_TASK:-16}" \
  --run-tag "${SLURM_JOB_NAME:-default}" \
  --case-tag harris_linear_20x6z \
  --timestep 1e-7 \
  --jacobian ./lin_ops.h5 \
  --massmat ./mass_mat.h5 \
  --cache-dir . \
  --output-dir .

# Plotting is a separate step and needs no HDF5, no operator and no cores, so
# run it wherever you read the results -- somewhere with network access, unless
# you pass --plot-inline-js:
#
#   nonmodal plot --output-dir . --min-level 1e-7 --nlevels 16
#
# The sampled region is always derived from the spectrum; widen or narrow it
# with --bounds-pad. Resolution comes from refinement rather than a fine initial
# lattice: 24x24 = 576 points to start, then 2600 more placed where the field
# actually varies, for ~3176 evaluations total.
