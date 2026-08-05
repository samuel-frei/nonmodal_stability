"""Fully resolved run configuration.

These carry no `None` and no `argparse.Namespace`: the CLI resolves everything
once, and the pipeline reads plain fields. Validation happens at construction,
so an invalid run fails before any matrix is loaded.
"""

from dataclasses import dataclass, field

from .sampling import PointSource


@dataclass(frozen=True)
class RunConfig:
  """Everything `nonmodal run` needs."""

  source: PointSource
  jacobian: str = './lin_ops.h5'
  massmat: str = './mass_mat.h5'
  cache_dir: str = '.'
  output_dir: str = 'pseudospectrum'
  nprocs: int = 128
  refine_rounds: int = 0
  #: Force full-plane sampling even when the operator is real. Half-plane
  #: sampling is otherwise chosen automatically from the operator itself.
  force_full_plane: bool = False
  run_tag: str = ''
  case_tag: str = ''

  def __post_init__(self) -> None:
    if self.nprocs < 1:
      raise ValueError('nprocs must be >= 1')
    if self.refine_rounds < 0:
      raise ValueError('refine-rounds must be >= 0')


@dataclass(frozen=True)
class PlotConfig:
  """Everything `nonmodal plot` needs.

  It reads a finished run directory, so it never touches HDF5 input, the
  operator, or the caches.
  """

  output_dir: str = 'pseudospectrum'
  plot_name: str = 'pseudoplot_default.html'
  nlevels: int = 16
  min_level: float = 1e-7
  #: Side length of the regular mesh the heatmap interpolates onto. Contours do
  #: not use it -- they are drawn from the samples themselves.
  mesh: int = 400
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
