"""Run metadata and array output."""

import argparse
import json
import os
import socket
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .pseudospectrum import _worker_count


def build_metadata(
  args: argparse.Namespace,
  real_min: float,
  real_max: float,
  imag_min: float,
  imag_max: float,
  grid_points: int,
  rows: int | None = None,
  cols: int | None = None,
  grid_type: str = 'structured',
  grid_source: str = 'generated',
) -> dict[str, Any]:
  """Build structured metadata describing one pseudospectrum run."""
  grid_points_effective = int(grid_points)
  effective_workers = _worker_count(grid_points_effective, args.nprocs)
  grid_shape = None
  if rows is not None and cols is not None:
    grid_shape = {
      'rows': int(rows),
      'cols': int(cols),
    }

  metadata = {
    'created_utc': datetime.now(UTC).isoformat(timespec='seconds').replace('+00:00', 'Z'),
    'hostname': socket.gethostname(),
    'cwd': os.getcwd(),
    'run_tag': args.run_tag,
    'case_tag': args.case_tag,
    'slurm_job_id': os.environ.get('SLURM_JOB_ID', ''),
    'inputs': {
      'jacobian': os.path.abspath(args.jacobian),
      'massmat': os.path.abspath(args.massmat),
      'cache_dir': os.path.abspath(args.cache_dir),
    },
    'grid_points': int(grid_points_effective),
    'grid_shape': grid_shape,
    'grid_type': grid_type,
    'grid_source': grid_source,
    'grid_npy': args.grid_npy or '',
    'nprocs_requested': int(args.nprocs),
    'nprocs_effective': int(effective_workers),
    'bounds': {
      'real_min': float(real_min),
      'real_max': float(real_max),
      'imag_min': float(imag_min),
      'imag_max': float(imag_max),
    },
    'levels': {
      'min_level': float(args.min_level),
      'nlevels': int(args.nlevels),
    },
  }

  return metadata


def write_metadata(output_dir: str, metadata: dict[str, Any]) -> None:
  """Persist run metadata as JSON for reproducibility and traceability."""
  path = os.path.join(output_dir, 'run_metadata.json')
  with open(path, 'w', encoding='ascii') as f:
    json.dump(metadata, f, indent=2, sort_keys=True)
  print(f'wrote metadata: {path}', flush=True)


def save_pseudospectrum_arrays(
  output_dir: str,
  R: NDArray[np.float64],
  C: NDArray[np.float64],
  sigmin: NDArray[np.float64],
) -> None:
  """Save the sampled grid and pseudospectrum values as NumPy arrays."""
  np.save(os.path.join(output_dir, 'pseudo_R.npy'), R)
  np.save(os.path.join(output_dir, 'pseudo_C.npy'), C)
  np.save(os.path.join(output_dir, 'pseudo_sigmin.npy'), sigmin)


def save_pseudospectrum_flat(
  output_dir: str,
  zz_flat: NDArray[np.complex128],
  sigmin_flat: NDArray[np.float64],
) -> None:
  """Save a flat complex grid and its pseudospectrum values as NumPy arrays."""
  np.save(os.path.join(output_dir, 'pseudo_z.npy'), zz_flat)
  np.save(os.path.join(output_dir, 'pseudo_sigmin_flat.npy'), sigmin_flat)
