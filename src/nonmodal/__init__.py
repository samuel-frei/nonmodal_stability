"""Nonmodal (pseudospectral) stability analysis of reduced linear MHD operators.

Loads exported Jacobian and mass matrices, reduces the global system to a
subset of physical fields, builds the Schur form of the resulting time-advance
operator, and samples the resolvent norm over the complex plane.

IMPORTANT: the BLAS thread limits below must be set before NumPy (or anything
importing it) is first imported, otherwise they have no effect. Sampling runs a
`fork` process pool, and unpinned BLAS threads would oversubscribe every core
against it -- which degrades many-core runs silently rather than erroring. Keep
these three assignments at the very top of this module, above every import.

`setdefault` rather than plain assignment: pinning to one thread is the right
default for the parallel sampling path, but a caller who deliberately sets a
thread count before importing (a notebook doing dense linear algebra, say) has
made an explicit choice we should not silently override.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from .fields import (  # noqa: E402
  FIELD_BLOCK_COUNT,
  FIELD_NAMES,
  KEPT_BLOCK_IDS,
  build_reduction_mapping,
  write_restart_eigenvectors,
)
from .grid import grid_bounds_from_flat, load_flat_grid_npy  # noqa: E402
from .io import (  # noqa: E402
  build_metadata,
  save_pseudospectrum_arrays,
  save_pseudospectrum_flat,
  write_metadata,
)
from .matrices import HDF5Matrix, assemble_global  # noqa: E402
from .operator import (  # noqa: E402
  DEFAULT_TIMESTEP,
  load_or_compute_eigvals,
  load_or_compute_jacobian,
  load_or_compute_schur,
)
from .pipeline import run_pipeline  # noqa: E402
from .plotting import pseudo_contours, pseudo_heatmap  # noqa: E402
from .pseudospectrum import (  # noqa: E402
  choose_contour_levels,
  compute_pseudospectrum,
)

__version__ = '0.1.0'

__all__ = [
  'DEFAULT_TIMESTEP',
  'FIELD_BLOCK_COUNT',
  'FIELD_NAMES',
  'HDF5Matrix',
  'KEPT_BLOCK_IDS',
  'assemble_global',
  'build_metadata',
  'build_reduction_mapping',
  'choose_contour_levels',
  'compute_pseudospectrum',
  'grid_bounds_from_flat',
  'load_flat_grid_npy',
  'load_or_compute_eigvals',
  'load_or_compute_jacobian',
  'load_or_compute_schur',
  'pseudo_contours',
  'pseudo_heatmap',
  'run_pipeline',
  'save_pseudospectrum_arrays',
  'save_pseudospectrum_flat',
  'write_metadata',
  'write_restart_eigenvectors',
]
