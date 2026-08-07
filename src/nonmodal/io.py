"""Run metadata and array output.

Sample sets are stored flat: `pseudo_z.npy` beside `pseudo_sigmin.npy`, with
`pseudo_eigvals.npy` alongside so `nonmodal plot` can draw its overlay without
reloading the operator.

* `save_samples` / `load_samples` -- the flat point set and its values.
* `build_metadata` -- describe one run, for reproducibility.
* `write_metadata` / `read_metadata` -- the run sidecar.
* `write_pseudomodes` / `read_pseudomodes` -- the pseudomode sidecar.
"""

import json
import os
import socket
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import RunConfig
from .sampling import Bounds

POINTS_FILE = 'pseudo_z.npy'
SIGMIN_FILE = 'pseudo_sigmin.npy'
EIGVALS_FILE = 'pseudo_eigvals.npy'
METADATA_FILE = 'run_metadata.json'
PSEUDOMODE_FILE = 'pseudomodes.json'


def build_metadata(
  config: RunConfig,
  bounds: Bounds,
  n_points: int,
  n_evaluated: int,
  half_plane: bool,
  effective_workers: int,
) -> dict[str, Any]:
  """Describe one run, for reproducibility and traceability."""
  return {
    'created_utc': datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': config.run_tag,
    'case_tag': config.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'inputs': {
      'jacobian': os.path.abspath(config.jacobian),
      'massmat': os.path.abspath(config.massmat),
      'cache_dir': os.path.abspath(config.cache_dir),
    },
    'sampling': {
      'source': config.source.describe(),
      'refine_rounds': int(config.refine_rounds),
      'points_total': int(n_points),
      'points_evaluated': int(n_evaluated),
      'half_plane_symmetry': bool(half_plane),
      'bounds': bounds.as_dict(),
    },
    'nprocs_requested': int(config.nprocs),
    'nprocs_effective': int(effective_workers),
  }


def _write_json(
  output_dir: str, filename: str, payload: dict[str, Any], label: str
) -> str:
  """Write one JSON sidecar, creating the directory if needed."""
  os.makedirs(output_dir, exist_ok=True)
  path = os.path.join(output_dir, filename)
  with open(path, 'w', encoding='ascii') as f:
    json.dump(payload, f, indent=2, sort_keys=True)
  print(f'wrote {label}: {path}', flush=True)
  return path


def _read_json(output_dir: str, filename: str) -> dict[str, Any]:
  with open(os.path.join(output_dir, filename), encoding='ascii') as f:
    return dict(json.load(f))


def write_metadata(output_dir: str, metadata: dict[str, Any]) -> None:
  """Persist run metadata as JSON."""
  _write_json(output_dir, METADATA_FILE, metadata, 'metadata')


def read_metadata(output_dir: str) -> dict[str, Any]:
  """Read back the metadata written by a run."""
  return _read_json(output_dir, METADATA_FILE)


def write_pseudomodes(output_dir: str, payload: dict[str, Any]) -> None:
  """Persist pseudomode provenance: where each mode came from and its residual."""
  _write_json(output_dir, PSEUDOMODE_FILE, payload, 'pseudomode metadata')


def read_pseudomodes(output_dir: str) -> dict[str, Any]:
  """Read back the pseudomode sidecar."""
  return _read_json(output_dir, PSEUDOMODE_FILE)


def save_samples(
  output_dir: str,
  z: NDArray[np.complex128],
  sigmin: NDArray[np.float64],
  eigvals: NDArray[np.complexfloating],
) -> None:
  """Save the sampled point set, its values, and the spectrum."""
  os.makedirs(output_dir, exist_ok=True)
  np.save(os.path.join(output_dir, POINTS_FILE), z)
  np.save(os.path.join(output_dir, SIGMIN_FILE), sigmin)
  np.save(os.path.join(output_dir, EIGVALS_FILE), eigvals)
  print(f'wrote {z.size} samples to {os.path.abspath(output_dir)}', flush=True)


def load_samples(
  output_dir: str,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complexfloating]]:
  """Load a finished run's samples, as written by `save_samples`."""
  z = np.load(os.path.join(output_dir, POINTS_FILE))
  sigmin = np.load(os.path.join(output_dir, SIGMIN_FILE))
  eigvals_path = os.path.join(output_dir, EIGVALS_FILE)
  eigvals = (
    np.load(eigvals_path)
    if os.path.exists(eigvals_path)
    else np.zeros(0, dtype=np.complex128))
  if z.shape != sigmin.shape:
    raise ValueError(
      f'{POINTS_FILE} and {SIGMIN_FILE} disagree: {z.shape} vs {sigmin.shape}')
  return z, sigmin, eigvals
