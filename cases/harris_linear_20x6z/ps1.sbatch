#!/usr/bin/env bash
#SBATCH --job-name=ps1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Required, not optional. Without it this cluster (select/cons_tres, CR_CORE)
# allocates a single core, SLURM_CPUS_PER_TASK goes unset, the fallback below
# forks 16 workers onto that one core, and nothing reports the mismatch --
# run_metadata.json logs the 16 it was asked for. Job 125399 lost ~8 hours to
# exactly this.
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00
#SBATCH --output=output_%x_%j.log
#SBATCH --error=error_%x_%j.log

# Samples a pseudospectrum and renders it, in one job.
#
# Submit from the directory holding lin_ops.h5 and mass_mat.h5 -- for the
# production case that is the `data/` directory, not the case directory:
#
#   cd /mnt/homes_global/swf2112/research/recon_data/psc_data/harris_linear_20x6z/data
#   sbatch /mnt/homes_global/swf2112/research/nonmodal/cases/harris_linear_20x6z/ps1.sbatch
#
# Caches land beside the exports (they are large: real_jacobian.npy is 882 MB and
# full_reduced_schur.npy 1.76 GB) and are reused by later jobs. Samples, metadata
# and plots land in ./pseudospectrum. That is the layout the Aug 2026 runs used.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- locate the `nonmodal` console script -------------------------------------
#
# sbatch copies this file into a spool directory, so $0 does not point back into
# the checkout at runtime and the repo cannot be found relative to this script.
# Resolution order: an already-activated environment, then the checkout's own
# .venv, then `uv run`. Override NONMODAL_REPO to point at a different checkout
# (export it before submitting; sbatch propagates the environment by default).
NONMODAL_REPO="${NONMODAL_REPO:-/mnt/homes_global/swf2112/research/nonmodal}"

if command -v nonmodal >/dev/null 2>&1; then
  RUN=(nonmodal)
elif [[ -x "${NONMODAL_REPO}/.venv/bin/nonmodal" ]]; then
  RUN=("${NONMODAL_REPO}/.venv/bin/nonmodal")
elif command -v uv >/dev/null 2>&1 && [[ -f "${NONMODAL_REPO}/pyproject.toml" ]]; then
  RUN=(uv run --project "${NONMODAL_REPO}" nonmodal)
else
  echo "ps1.sbatch: cannot find the 'nonmodal' console script." >&2
  echo "  Tried: PATH, ${NONMODAL_REPO}/.venv/bin/nonmodal, uv run --project ${NONMODAL_REPO}" >&2
  echo "  Set NONMODAL_REPO to the checkout, or activate an environment providing it." >&2
  exit 1
fi
echo "ps1.sbatch: using ${RUN[*]}"

# Fail before the allocation is spent rather than several minutes in.
for f in lin_ops.h5 mass_mat.h5; do
  [[ -f "$f" ]] || { echo "ps1.sbatch: $PWD/$f not found -- submit from the export directory." >&2; exit 1; }
done

# operator.py imports pyplot at module scope; compute nodes have no display.
export MPLBACKEND=Agg

# No srun: a single-task job's batch step already has the whole allocation, so
# srun adds a step for nothing. (This cluster runs SLURM 21.08, where srun does
# still inherit --cpus-per-task; 22.05 dropped that, so avoiding srun keeps the
# deck correct on both.) The package pins BLAS to one thread per worker itself.
#
# The fallback is a backstop, not a default: if it ever fires, --cpus-per-task is
# missing above and the pool is oversubscribing one core.
NPROCS="${SLURM_CPUS_PER_TASK:-16}"
if [[ -z "${SLURM_CPUS_PER_TASK:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  echo "ps1.sbatch: SLURM_CPUS_PER_TASK unset under a job allocation --" >&2
  echo "  --cpus-per-task is missing, so ${NPROCS} workers would share one core." >&2
  exit 1
fi

"${RUN[@]}" run \
  --grid-nx 25 --grid-ny 25 \
  --refine-points 2600 \
  --nprocs "${NPROCS}" \
  --run-tag "${SLURM_JOB_NAME:-default}" \
  --case-tag harris_linear_20x6z \
  --timestep 1e-7 \
  --jacobian ./lin_ops.h5 \
  --massmat ./mass_mat.h5 \
  --cache-dir . \
  --output-dir ./pseudospectrum

# Plotting needs no HDF5, no operator and no cores -- it reads the flat samples
# and pseudo_eigvals.npy back from --output-dir. It runs here so the job is
# self-contained; --plot-inline-js embeds plotly.js, since a page that links the
# CDN renders blank on a node without network access. If this step fails the
# samples are already on disk: rerun just this command anywhere.
"${RUN[@]}" plot \
  --output-dir ./pseudospectrum \
  --min-level 1e-7 \
  --nlevels 16 \
  --plot-inline-js

# The sampled region is always derived from the spectrum; widen or narrow it with
# --bounds-pad. Resolution comes from refinement rather than a fine initial
# lattice: a 25x25 lattice to start -- of which only the upper half is evaluated,
# since the operator is real -- then up to 2600 more points placed where the field
# actually varies, spread over --refine-rounds (4 by default). Im z = 0 is always
# one of the sampled rows.
#
# The caches are keyed by filename alone. After changing --timestep, the exports,
# or KEPT_BLOCK_IDS, delete real_jacobian.npy and full_reduced_*.npy by hand --
# nothing invalidates them for you.
