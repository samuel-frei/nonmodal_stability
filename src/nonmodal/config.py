"""Fully resolved run configuration.

These carry no `None` and no `argparse.Namespace`: the CLI resolves everything
once, and the pipeline reads plain fields. Validation happens at construction,
so an invalid run fails before any matrix is loaded.
"""

from dataclasses import dataclass, field

from .operator import DEFAULT_TIMESTEP
from .sampling import PointSource

DEFAULT_N_EIGVECS = 40
DEFAULT_REFINE_ROUNDS = 4
DEFAULT_NLEVELS = 16
DEFAULT_MIN_LEVEL = 1e-7
DEFAULT_PLOT_MESH = 400


@dataclass(frozen=True)
class RunConfig:
  """Everything `nonmodal run` needs."""

  source: PointSource
  jacobian: str = './lin_ops.h5'
  massmat: str = './mass_mat.h5'
  cache_dir: str = '.'
  output_dir: str = 'pseudospectrum'
  nprocs: int = 128
  #: Ceiling on extra evaluations refinement may spend on top of the initial
  #: grid; zero disables it. Not a guarantee -- see `refine.refine`.
  refine_points: int = 0
  #: How many rounds to spread `refine_points` over. More rounds adapt more
  #: closely, at one worker-pool round trip each.
  refine_rounds: int = DEFAULT_REFINE_ROUNDS
  #: Force full-plane sampling even when the operator is real. Half-plane
  #: sampling is otherwise chosen automatically from the operator itself.
  force_full_plane: bool = False
  #: Timestep of the implicit solve that produced the Jacobian. It defines the
  #: effective operator, so it must match the simulation that exported it.
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
class PlotConfig:
  """Everything `nonmodal plot` needs.

  It reads a finished run directory, so it never touches HDF5 input, the
  operator, or the caches.
  """

  output_dir: str = 'pseudospectrum'
  plot_name: str = 'pseudoplot_default.html'
  nlevels: int = DEFAULT_NLEVELS
  min_level: float = DEFAULT_MIN_LEVEL
  #: Side length of the regular mesh the heatmap interpolates onto. Contours do
  #: not use it -- they are drawn from the samples themselves.
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
