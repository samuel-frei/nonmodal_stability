#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/lin_ops_full_reduction.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "ERROR: Missing ${PY_SCRIPT}" >&2
  exit 1
fi

OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/outputs}"
EXPECTED_RUN_DB_HEADER="submit_utc,run_tag,case_tag,job_id,job_name,state,dependency_job_id,grid_points,nprocs,cpus_per_task,time_limit,partition,real_min,real_max,imag_min,imag_max,output_dir,job_script,stdout_log,stderr_log"

if [[ $# -gt 0 ]]; then
  CASE_TAGS=("$@")
else
  CASE_TAGS=(default)
fi

NODE_CORES="${NODE_CORES:-128}"
GRID_POINTS="${GRID_POINTS:-${NODE_CORES}}"
NPROCS="${NPROCS:-${NODE_CORES:-128}}"
BLOCK_FACTOR="${BLOCK_FACTOR:-4}"
MIN_LEVEL="${MIN_LEVEL:-1e-8}"
NLEVELS="${NLEVELS:-8}"
REAL_MIN="${REAL_MIN:-0.1}"
REAL_MAX="${REAL_MAX:-2.0}"
IMAG_MIN="${IMAG_MIN:--0.3}"
IMAG_MAX="${IMAG_MAX:-0.3}"
REAL_SWEEP_WINDOWS="${REAL_SWEEP_WINDOWS:-}"

# Slurm controls (override via environment variables).
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_TIME="${SLURM_TIME:-0-04:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-${NODE_CORES}}"

if (( SLURM_CPUS_PER_TASK < NODE_CORES )); then
  echo "Adjusting SLURM_CPUS_PER_TASK from ${SLURM_CPUS_PER_TASK} to ${NODE_CORES} (full-node billing policy)"
  SLURM_CPUS_PER_TASK="${NODE_CORES}"
fi

# Always maximize usage of allocated cores by matching worker count to cpus-per-task.
if [[ "${NPROCS}" != "${SLURM_CPUS_PER_TASK}" ]]; then
  echo "Adjusting NPROCS from ${NPROCS} to ${SLURM_CPUS_PER_TASK} to match cpus-per-task"
  NPROCS="${SLURM_CPUS_PER_TASK}"
fi

# Enforce minimum grid size based on node core count policy.
if (( GRID_POINTS < NODE_CORES )); then
  echo "Adjusting GRID_POINTS from ${GRID_POINTS} to ${NODE_CORES} (minimum grid policy)"
  GRID_POINTS="${NODE_CORES}"
fi

if (( BLOCK_FACTOR < 1 )); then
  echo "ERROR: BLOCK_FACTOR must be >= 1" >&2
  exit 1
fi

# Enforce equal per-process work while preserving requested worker/core count.
# Round GRID_POINTS up to the next multiple of NPROCS.
if (( GRID_POINTS % NPROCS != 0 )); then
  OLD_GRID_POINTS="${GRID_POINTS}"
  GRID_POINTS=$(( ((GRID_POINTS + NPROCS - 1) / NPROCS) * NPROCS ))
  echo "Adjusting GRID_POINTS from ${OLD_GRID_POINTS} to ${GRID_POINTS} so work is evenly divisible across ${NPROCS} workers"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TAG="ps_${STAMP}"
OUT_DIR="${OUT_BASE}/${OUT_TAG}"
mkdir -p "${OUT_DIR}"
RUN_DB="${OUT_BASE}/run_db.csv"

if [[ ! -f "${RUN_DB}" ]]; then
  echo "${EXPECTED_RUN_DB_HEADER}" > "${RUN_DB}"
else
  RUN_DB_HEADER="$(head -n 1 "${RUN_DB}")"
  if [[ "${RUN_DB_HEADER}" != "${EXPECTED_RUN_DB_HEADER}" ]]; then
    LEGACY_HEADER="submit_utc,run_tag,case_tag,job_id,job_name,state,grid_points,nprocs,cpus_per_task,time_limit,partition,real_min,real_max,imag_min,imag_max,output_dir,job_script,stdout_log,stderr_log"
    if [[ "${RUN_DB_HEADER}" == "${LEGACY_HEADER}" ]]; then
      TMP_RUN_DB="$(mktemp "${OUT_BASE}/run_db_upgrade.XXXXXX")"
      awk -F',' 'BEGIN { OFS="," }
        NR==1 {
          print "submit_utc,run_tag,case_tag,job_id,job_name,state,dependency_job_id,grid_points,nprocs,cpus_per_task,time_limit,partition,real_min,real_max,imag_min,imag_max,output_dir,job_script,stdout_log,stderr_log"
          next
        }
        {
          print $1,$2,$3,$4,$5,$6,"",$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19
        }
      ' "${RUN_DB}" > "${TMP_RUN_DB}"
      mv "${TMP_RUN_DB}" "${RUN_DB}"
      echo "Upgraded run database schema to include dependency_job_id"
    else
      echo "ERROR: Unrecognized run database header in ${RUN_DB}" >&2
      echo "ERROR: Header found: ${RUN_DB_HEADER}" >&2
      exit 1
    fi
  fi
fi

SWEEP_ENABLED=0
SWEEP_COUNT=0
declare -a SWEEP_REAL_MINS=()
declare -a SWEEP_REAL_MAXS=()

if [[ -n "${REAL_MIN}" || -n "${REAL_MAX}" || -n "${IMAG_MIN}" || -n "${IMAG_MAX}" ]]; then
  if [[ -z "${REAL_MIN}" || -z "${REAL_MAX}" || -z "${IMAG_MIN}" || -z "${IMAG_MAX}" ]]; then
    echo "ERROR: Set all REAL_MIN REAL_MAX IMAG_MIN IMAG_MAX together" >&2
    exit 1
  fi
fi

if [[ -n "${REAL_SWEEP_WINDOWS}" ]]; then
  SWEEP_ENABLED=1
  if [[ -z "${IMAG_MIN}" || -z "${IMAG_MAX}" ]]; then
    echo "ERROR: Set IMAG_MIN and IMAG_MAX when REAL_SWEEP_WINDOWS is used" >&2
    exit 1
  fi

  IFS=',' read -r -a _RAW_WINDOWS <<< "${REAL_SWEEP_WINDOWS}"
  for _win in "${_RAW_WINDOWS[@]}"; do
    _win_trimmed="${_win//[[:space:]]/}"
    if [[ -z "${_win_trimmed}" ]]; then
      continue
    fi
    if [[ "${_win_trimmed}" != *:* ]]; then
      echo "ERROR: REAL_SWEEP_WINDOWS entries must be min:max (bad entry: ${_win})" >&2
      exit 1
    fi
    _sweep_min="${_win_trimmed%%:*}"
    _sweep_max="${_win_trimmed##*:}"
    if [[ -z "${_sweep_min}" || -z "${_sweep_max}" ]]; then
      echo "ERROR: REAL_SWEEP_WINDOWS entries must include both min and max (bad entry: ${_win})" >&2
      exit 1
    fi
    SWEEP_REAL_MINS+=("${_sweep_min}")
    SWEEP_REAL_MAXS+=("${_sweep_max}")
  done

  SWEEP_COUNT="${#SWEEP_REAL_MINS[@]}"
  if (( SWEEP_COUNT == 0 )); then
    echo "ERROR: REAL_SWEEP_WINDOWS is set but no valid windows were parsed" >&2
    exit 1
  fi

  if (( $# == 0 )); then
    CASE_TAGS=()
    for (( i=0; i<SWEEP_COUNT; i++ )); do
      CASE_TAGS+=("re_${SWEEP_REAL_MINS[$i]}_${SWEEP_REAL_MAXS[$i]}")
    done
  else
    if (( ${#CASE_TAGS[@]} != SWEEP_COUNT )); then
      echo "ERROR: Number of CASE_TAGS (${#CASE_TAGS[@]}) must match number of REAL_SWEEP_WINDOWS entries (${SWEEP_COUNT})" >&2
      exit 1
    fi
  fi
fi

TOTAL="${#CASE_TAGS[@]}"
COUNT=0

echo "Batch pseudospectrum run starting"
echo "Output root: ${OUT_DIR}"
echo "Runs: ${TOTAL}"
echo "Submitting with Slurm via sbatch"
echo "Dependency chaining: sequential afterok"
echo "Python executable: ${PYTHON_BIN}"
if [[ -n "${REAL_MIN}" || -n "${REAL_MAX}" || -n "${IMAG_MIN}" || -n "${IMAG_MAX}" ]]; then
  echo "Explicit grid bounds requested"
fi
if (( SWEEP_ENABLED == 1 )); then
  echo "Real-axis sweep requested via REAL_SWEEP_WINDOWS (${SWEEP_COUNT} windows)"
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found in PATH" >&2
  exit 1
fi

JOB_SCRIPT_DIR="${OUT_DIR}/job_scripts"
mkdir -p "${JOB_SCRIPT_DIR}"

PREV_JOB_ID="$(awk -F',' '
  NR==1 {
    for (i=1; i<=NF; i++) {
      if ($i == "job_id") {
        job_idx=i
      }
    }
    next
  }
  {
    if (job_idx > 0 && $job_idx != "") {
      last_job_id=$job_idx
    }
  }
  END {
    if (last_job_id != "") {
      print last_job_id
    }
  }
' "${RUN_DB}")"

if [[ -n "${PREV_JOB_ID}" ]]; then
  echo "Cross-invocation chaining enabled: first job will depend on job ${PREV_JOB_ID}"
else
  echo "Cross-invocation chaining enabled: no prior job found, first job has no dependency"
fi

for CASE_TAG in "${CASE_TAGS[@]}"; do
  COUNT=$((COUNT + 1))

  CASE_REAL_MIN="${REAL_MIN}"
  CASE_REAL_MAX="${REAL_MAX}"
  CASE_IMAG_MIN="${IMAG_MIN}"
  CASE_IMAG_MAX="${IMAG_MAX}"
  if (( SWEEP_ENABLED == 1 )); then
    SWEEP_IDX=$((COUNT - 1))
    CASE_REAL_MIN="${SWEEP_REAL_MINS[$SWEEP_IDX]}"
    CASE_REAL_MAX="${SWEEP_REAL_MAXS[$SWEEP_IDX]}"
  fi

  SAFE_TAG="${CASE_TAG//./p}"
  SAFE_TAG="${SAFE_TAG//-/m}"
  SAFE_TAG="${SAFE_TAG//+/}"

  RUN_OUT="${OUT_DIR}/c${COUNT}_${SAFE_TAG}"
  PLOT_NAME="pseudoplot_${SAFE_TAG}.html"
  mkdir -p "${RUN_OUT}"

  JOB_NAME="ps${COUNT}"
  LOG_OUT="${RUN_OUT}/output_${SAFE_TAG}.log"
  LOG_ERR="${RUN_OUT}/error_${SAFE_TAG}.log"
  JOB_SCRIPT="${JOB_SCRIPT_DIR}/${JOB_NAME}.sbatch"

  if [[ -n "${CASE_REAL_MIN}" ]]; then
    GRID_BOUNDS="--real-min ${CASE_REAL_MIN} --real-max ${CASE_REAL_MAX} --imag-min ${CASE_IMAG_MIN} --imag-max ${CASE_IMAG_MAX}"
  else
    GRID_BOUNDS=""
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=${JOB_NAME}"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=${SLURM_CPUS_PER_TASK}"
    echo "#SBATCH --time=${SLURM_TIME}"
    echo "#SBATCH --output=${LOG_OUT}"
    echo "#SBATCH --error=${LOG_ERR}"
    if [[ -n "${SLURM_PARTITION}" ]]; then
      echo "#SBATCH --partition=${SLURM_PARTITION}"
    fi
    if [[ -n "${SLURM_ACCOUNT}" ]]; then
      echo "#SBATCH --account=${SLURM_ACCOUNT}"
    fi
    echo "#SBATCH --mail-user=swf2112@columbia.edu"
    echo "#SBATCH --mail-type=ALL"
    cat <<EOF

source /jet/home/freiberg/.bashrc
module load anaconda3
conda activate openfusion

set -euo pipefail

echo "Starting case=${CASE_TAG} at \\$(date)"
srun ${PYTHON_BIN} ${PY_SCRIPT} \\
  --grid-points ${GRID_POINTS} \\
  --nprocs ${NPROCS} \\
  --block-factor ${BLOCK_FACTOR} \\
  --min-level ${MIN_LEVEL} \\
  --nlevels ${NLEVELS} \\
  --run-tag ${OUT_TAG} \\
  --case-tag ${CASE_TAG} \\
  ${GRID_BOUNDS} \\
  --output-dir ${RUN_OUT} \\
  --plot-name ${PLOT_NAME}
echo "Finished case=${CASE_TAG} at \\$(date)"
EOF
  } > "${JOB_SCRIPT}"

  chmod +x "${JOB_SCRIPT}"

  echo "[$COUNT/$TOTAL] Submitting case=${CASE_TAG}"
  DEPENDENCY_JOB_ID=""
  SBATCH_ARGS=("--parsable" "${JOB_SCRIPT}")
  if [[ -n "${PREV_JOB_ID}" ]]; then
    DEPENDENCY_JOB_ID="${PREV_JOB_ID}"
    SBATCH_ARGS=("--dependency=afterok:${PREV_JOB_ID}" "${SBATCH_ARGS[@]}")
  fi
  JOB_ID="$(sbatch "${SBATCH_ARGS[@]}")"
  echo "[$COUNT/$TOTAL] Submitted job ${JOB_ID} for case=${CASE_TAG}"
  SUBMIT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  PARTITION_VALUE="${SLURM_PARTITION}"
  if [[ -z "${PARTITION_VALUE}" ]]; then
    PARTITION_VALUE="(default)"
  fi
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${SUBMIT_UTC}" "${OUT_TAG}" "${CASE_TAG}" "${JOB_ID}" "${JOB_NAME}" "SUBMITTED" "${DEPENDENCY_JOB_ID}" \
    "${GRID_POINTS}" "${NPROCS}" "${SLURM_CPUS_PER_TASK}" "${SLURM_TIME}" "${PARTITION_VALUE}" \
    "${CASE_REAL_MIN}" "${CASE_REAL_MAX}" "${CASE_IMAG_MIN}" "${CASE_IMAG_MAX}" \
    "${RUN_OUT}" "${JOB_SCRIPT}" "${LOG_OUT}" "${LOG_ERR}" >> "${RUN_DB}"
  PREV_JOB_ID="${JOB_ID}"
done

echo "All jobs submitted"
echo "Batch directory: ${OUT_DIR}"
echo "Run database: ${RUN_DB}"
echo "Use: squeue -u \"$USER\" to monitor jobs"
