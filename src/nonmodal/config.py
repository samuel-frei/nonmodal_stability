"""Fully resolved run configuration.

No `None` and no `argparse.Namespace`: the CLI resolves everything once and the
pipeline reads plain fields. Validation happens at construction, so an invalid
run fails before any matrix is loaded.

* `RunConfig` -- everything `nonmodal run` needs.
* `PseudomodeConfig` -- everything `nonmodal pseudomode` needs.
* `PlotConfig` -- everything `nonmodal plot` needs.
"""

from dataclasses import dataclass, field

from .operator import DEFAULT_TIMESTEP
from .pseudospectrum import DEFAULT_MODE_TOL
from .sampling import PointSource

DEFAULT_N_EIGVECS = 40
DEFAULT_REFINE_ROUNDS = 4
DEFAULT_NLEVELS = 16
DEFAULT_MIN_LEVEL = 1e-7
DEFAULT_PLOT_MESH = 400
DEFAULT_MODE_DIR = 'pseudomodes'


@dataclass(frozen=True)
class RunConfig:
  """Everything `nonmodal run` needs."""

  source: PointSource
  jacobian: str = './lin_ops.h5'
  massmat: str = './mass_mat.h5'
  cache_dir: str = '.'
  output_dir: str = 'pseudospectrum'
  nprocs: int = 128
  #: Ceiling on extra evaluations refinement may spend; 0 disables it.
  refine_points: int = 0
  #: Rounds to spread `refine_points` over, one worker-pool trip each.
  refine_rounds: int = DEFAULT_REFINE_ROUNDS
  #: Force full-plane sampling; otherwise decided from the operator itself.
  force_full_plane: bool = False
  #: Timestep of the implicit solve; it defines the effective operator.
  timestep: float = DEFAULT_TIMESTEP
  #: How many eigenvectors to compute and write out as restart files.
  n_eigvecs: int = DEFAULT_N_EIGVECS
  run_tag: str = ''
  case_tag: str = ''

  def __post_init__(self) -> None:
    if self.nprocs < 1:
      raise ValueError('nprocs must be >= 1')
    if self.refine_points < 0:
      raise ValueError('refine-points must be >= 0')
    if self.refine_rounds < 1:
      raise ValueError('refine-rounds must be >= 1')
    if self.timestep <= 0.0:
      raise ValueError('timestep must be positive')
    if self.n_eigvecs < 1:
      raise ValueError('n-eigvecs must be >= 1')


@dataclass(frozen=True)
class PseudomodeConfig:
  """Everything `nonmodal pseudomode` needs.

  A compute command, but each point `z` is given rather than searched for.
  """

  #: The points to extract modes at, given explicitly.
  points: tuple[complex, ...] = ()
  jacobian: str = './lin_ops.h5'
  massmat: str = './mass_mat.h5'
  cache_dir: str = '.'
  output_dir: str = 'pseudospectrum'
  #: Subdirectory for the restarts, kept apart from `eigvecs_plot/`.
  mode_dir: str = DEFAULT_MODE_DIR
  #: Restart files per mode; 1 is the amplitude-maximising phase.
  phases: int = 1
  tol: float = DEFAULT_MODE_TOL
  timestep: float = DEFAULT_TIMESTEP
  run_tag: str = ''
  case_tag: str = ''

  def __post_init__(self) -> None:
    if not self.points:
      raise ValueError('at least one --at point is required')
    if self.phases < 1:
      raise ValueError('phases must be >= 1')
    if self.tol <= 0.0:
      raise ValueError('mode-tol must be positive')
    if self.timestep <= 0.0:
      raise ValueError('timestep must be positive')


@dataclass(frozen=True)
class PlotConfig:
  """Everything `nonmodal plot` needs.

  Reads a finished run directory: no HDF5, no operator, no caches.
  """

  output_dir: str = 'pseudospectrum'
  plot_name: str = 'pseudoplot_default.html'
  nlevels: int = DEFAULT_NLEVELS
  min_level: float = DEFAULT_MIN_LEVEL
  #: Side length of the heatmap's mesh; contours do not use it.
  mesh: int = DEFAULT_PLOT_MESH
  #: Embed plotly.js instead of linking the CDN, so plots render offline.
  inline_js: bool = False
  levels: tuple[float, ...] = field(default_factory=tuple)

  def __post_init__(self) -> None:
    if self.nlevels < 1:
      raise ValueError('nlevels must be >= 1')
    if self.min_level <= 0.0:
      raise ValueError('min-level must be positive')
    if self.mesh < 2:
      raise ValueError('plot-mesh must be >= 2')
