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

from .config import PlotConfig, RunConfig  # noqa: E402
from .fields import (  # noqa: E402
  FIELD_BLOCK_COUNT,
  FIELD_NAMES,
  KEPT_BLOCK_IDS,
  build_reduction_mapping,
  write_restart_eigenvectors,
)
from .io import (  # noqa: E402
  build_metadata,
  load_samples,
  read_metadata,
  save_samples,
  write_metadata,
)
from .matrices import HDF5Matrix, assemble_global  # noqa: E402
from .operator import (  # noqa: E402
  DEFAULT_TIMESTEP,
  load_or_compute_eigvals,
  load_or_compute_jacobian,
  load_or_compute_schur,
)
from .pipeline import plot_run, run_pipeline  # noqa: E402
from .plotting import interpolate_to_mesh, pseudo_contours, pseudo_heatmap  # noqa: E402
from .pseudospectrum import (  # noqa: E402
  choose_contour_levels,
  compute_pseudospectrum,
  sample_sigmin,
)
from .refine import refine, triangle_errors  # noqa: E402
from .sampling import (  # noqa: E402
  Bounds,
  FileSource,
  RectangularSource,
  SpectrumSource,
  load_flat_grid_npy,
  mirror_conjugates,
  near_square,
  uniform_points,
)

__version__ = '0.1.0'

__all__ = [
  'DEFAULT_TIMESTEP',
  'FIELD_BLOCK_COUNT',
  'FIELD_NAMES',
  'KEPT_BLOCK_IDS',
  'Bounds',
  'FileSource',
  'HDF5Matrix',
  'PlotConfig',
  'RectangularSource',
  'SpectrumSource',
  'RunConfig',
  'assemble_global',
  'build_metadata',
  'build_reduction_mapping',
  'choose_contour_levels',
  'compute_pseudospectrum',
  'interpolate_to_mesh',
  'load_flat_grid_npy',
  'load_or_compute_eigvals',
  'load_or_compute_jacobian',
  'load_or_compute_schur',
  'load_samples',
  'mirror_conjugates',
  'near_square',
  'plot_run',
  'pseudo_contours',
  'pseudo_heatmap',
  'read_metadata',
  'refine',
  'run_pipeline',
  'sample_sigmin',
  'save_samples',
  'triangle_errors',
  'uniform_points',
  'write_metadata',
  'write_restart_eigenvectors',
]
