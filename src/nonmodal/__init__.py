"""Nonmodal (pseudospectral) stability analysis of reduced linear MHD operators.

Loads exported Jacobian and mass matrices, reduces the system to a subset of
physical fields, builds the Schur form of the time-advance operator, and samples
the resolvent norm over the complex plane.

Module map: `matrices` and `fields` read the exports, `operator` builds and
caches the reduced operator, `sampling` and `refine` decide where to evaluate,
`pseudospectrum` evaluates, `pseudomode` extracts a single mode, `plotting`
renders, and `config`/`cli`/`pipeline` tie them together.

IMPORTANT: the BLAS thread limits below take effect only when set before NumPy
is first imported, so they stay above every import here -- which is what the
`# noqa: E402` markers record. Sampling runs a `fork` pool, which one BLAS
thread per worker keeps from oversubscribing. `setdefault` leaves the choice to
a caller who sets threads before importing.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from .adapters import DofCoordinates, read_dof_coordinates  # noqa: E402
from .config import (  # noqa: E402
  FluxConfig,
  KreissConfig,
  PlotConfig,
  PseudomodeConfig,
  RunConfig,
)
from .fields import (  # noqa: E402
  FIELD_BLOCK_COUNT,
  FIELD_NAMES,
  KEPT_BLOCK_IDS,
  aligned_phase,
  build_reduction_mapping,
  write_restart_eigenvectors,
  write_restart_modes,
)
from .flux import (  # noqa: E402
  FluxFunctional,
  FluxSeed,
  adjoint_vector,
  flux_gain_curve,
  flux_optimal_seed,
  flux_response,
  sheet_flux_functional,
  sheet_flux_weights,
)
from .io import (  # noqa: E402
  build_metadata,
  load_samples,
  read_flux,
  read_kreiss,
  read_metadata,
  read_pseudomodes,
  save_samples,
  write_flux,
  write_kreiss,
  write_metadata,
  write_pseudomodes,
)
from .kreiss import KreissResult, kreiss_constant  # noqa: E402
from .matrices import HDF5Matrix, assemble_global  # noqa: E402
from .norm import (  # noqa: E402
  L2_COEFFICIENTS,
  Norm,
  norm_matrix,
  scaled_l2_norm,
  weighted_operator,
)
from .operator import (  # noqa: E402
  DEFAULT_TIMESTEP,
  eigenvectors_from_schur,
  load_or_compute_generator,
  load_or_compute_schur,
  load_or_compute_schur_vectors,
  rightmost_indices,
  spectrum_from_schur,
  write_eigenmode_restarts,
)
from .pipeline import flux_run, kreiss_run, plot_run, pseudomode_run, run_pipeline  # noqa: E402
from .plotting import interpolate_to_mesh, pseudo_contours, pseudo_heatmap  # noqa: E402
from .pseudomode import Pseudomode, pseudomode_at  # noqa: E402
from .pseudospectrum import (  # noqa: E402
  choose_contour_levels,
  compute_pseudospectrum,
  sample_sigmin,
  sigmin_with_mode,
)
from .refine import refine, triangle_errors  # noqa: E402
from .sampling import (  # noqa: E402
  Bounds,
  FileSource,
  RectangularSource,
  SpectrumSource,
  load_flat_grid_npy,
  mirror_conjugates,
  uniform_points,
)

__version__ = '0.1.0'

__all__ = [
  'Bounds',
  'DEFAULT_TIMESTEP',
  'DofCoordinates',
  'FIELD_BLOCK_COUNT',
  'FIELD_NAMES',
  'FileSource',
  'FluxConfig',
  'FluxFunctional',
  'FluxSeed',
  'HDF5Matrix',
  'KEPT_BLOCK_IDS',
  'KreissConfig',
  'KreissResult',
  'PlotConfig',
  'Pseudomode',
  'PseudomodeConfig',
  'RectangularSource',
  'RunConfig',
  'SpectrumSource',
  'adjoint_vector',
  'aligned_phase',
  'assemble_global',
  'build_metadata',
  'build_reduction_mapping',
  'choose_contour_levels',
  'compute_pseudospectrum',
  'flux_gain_curve',
  'flux_optimal_seed',
  'flux_response',
  'flux_run',
  'interpolate_to_mesh',
  'kreiss_constant',
  'L2_COEFFICIENTS',
  'Norm',
  'norm_matrix',
  'scaled_l2_norm',
  'weighted_operator',
  'kreiss_run',
  'load_flat_grid_npy',
  'eigenvectors_from_schur',
  'rightmost_indices',
  'spectrum_from_schur',
  'write_eigenmode_restarts',
  'load_or_compute_generator',
  'load_or_compute_schur',
  'load_or_compute_schur_vectors',
  'load_samples',
  'mirror_conjugates',
  'plot_run',
  'pseudo_contours',
  'pseudo_heatmap',
  'pseudomode_at',
  'pseudomode_run',
  'read_dof_coordinates',
  'read_flux',
  'read_kreiss',
  'read_metadata',
  'read_pseudomodes',
  'refine',
  'run_pipeline',
  'sample_sigmin',
  'save_samples',
  'sheet_flux_functional',
  'sheet_flux_weights',
  'sigmin_with_mode',
  'triangle_errors',
  'uniform_points',
  'write_flux',
  'write_kreiss',
  'write_metadata',
  'write_pseudomodes',
  'write_restart_eigenvectors',
  'write_restart_modes',
]
