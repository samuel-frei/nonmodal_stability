"""Command-line interface.

Two subcommands, because computing and looking at the result have different
requirements: `run` needs the HDF5 matrices and a machine with cores, `plot`
needs only a finished output directory.

This is the only module that touches `argparse.Namespace`; everything below it
receives a resolved `RunConfig` or `PlotConfig`.

Flags that cannot take effect are rejected rather than ignored. Silently
accepting `--refine-rounds` alongside `--grid-npy`, say, would let a two-day job
run without the behaviour it was asked for.
"""

import argparse

from .config import (
  DEFAULT_MIN_LEVEL,
  DEFAULT_N_EIGVECS,
  DEFAULT_NLEVELS,
  PlotConfig,
  RunConfig,
)
from .operator import DEFAULT_TIMESTEP
from .pipeline import plot_run, run_pipeline
from .refine import DEFAULT_SEED_FRACTION
from .sampling import (
  DEFAULT_BOUNDS_PAD,
  Bounds,
  FileSource,
  PointSource,
  RectangularSource,
  SpectrumSource,
  near_square,
)

BOUND_FLAGS = ('real_min', 'real_max', 'imag_min', 'imag_max')


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
  grid = parser.add_argument_group('sampling region')
  grid.add_argument('--grid-points', type=int, default=128,
                    help='Total number of sigma_min evaluations, split into a '
                         'near-square lattice. Use --grid-nx/--grid-ny for an '
                         'explicit shape.')
  grid.add_argument('--grid-nx', type=int, default=None,
                    help='Lattice columns; overrides --grid-points.')
  grid.add_argument('--grid-ny', type=int, default=None,
                    help='Lattice rows; overrides --grid-points.')
  grid.add_argument('--real-min', type=float, default=None,
                    help='Sampled real-axis minimum. Omit all four bounds to '
                         'infer the region from the spectrum.')
  grid.add_argument('--real-max', type=float, default=None,
                    help='Sampled real-axis maximum.')
  grid.add_argument('--imag-min', type=float, default=None,
                    help='Sampled imaginary-axis minimum.')
  grid.add_argument('--imag-max', type=float, default=None,
                    help='Sampled imaginary-axis maximum.')
  grid.add_argument('--bounds-pad', type=float, default=DEFAULT_BOUNDS_PAD,
                    help='Padding around the spectrum when bounds are inferred, '
                         'as a fraction of the spectral span.')
  grid.add_argument('--grid-npy', type=str, default='',
                    help='Optional .npy file of flat complex points to sample, '
                         'instead of a rectangular region.')

  refine = parser.add_argument_group('adaptive refinement')
  refine.add_argument('--refine-rounds', type=int, default=0,
                      help='Refinement rounds (0 disables it, the default). '
                           'Measurably beats uniform sampling on strongly '
                           'non-normal operators, roughly a wash otherwise.')
  refine.add_argument('--refine-seed-fraction', type=float,
                      default=DEFAULT_SEED_FRACTION,
                      help='Fraction of the budget spent on the initial seed.')

  op = parser.add_argument_group('operator')
  op.add_argument('--jacobian', type=str, default='./lin_ops.h5',
                  help='HDF5 file holding the /jacobian matrix.')
  op.add_argument('--massmat', type=str, default='./mass_mat.h5',
                  help='HDF5 file holding the /massmat matrix.')
  op.add_argument('--timestep', type=float, default=None,
                  help='Timestep of the implicit solve that produced the '
                       'Jacobian. It defines the effective operator, so it must '
                       'match the simulation that exported it.')
  op.add_argument('--n-eigvecs', type=int, default=None,
                  help='How many eigenvectors to compute and write as restart '
                       'files.')
  op.add_argument('--no-half-plane', action='store_true',
                  help='Sample the full plane even for a real operator, instead '
                       'of halving the work via conjugate symmetry.')

  out = parser.add_argument_group('output')
  out.add_argument('--nprocs', type=int, default=128, help='Worker process count.')
  out.add_argument('--cache-dir', type=str, default='.',
                   help='Directory for the reduced-operator, eigenvalue and Schur '
                        'caches. Caches are reused by filename alone and are NOT '
                        'invalidated when inputs or --timestep change; delete them '
                        'by hand after changing either.')
  out.add_argument('--output-dir', type=str, default='pseudospectrum',
                   help='Output directory for samples and metadata.')
  out.add_argument('--run-tag', type=str, default='',
                   help='Batch-level run identifier for metadata tracking.')
  out.add_argument('--case-tag', type=str, default='',
                   help='Case identifier for metadata tracking.')


def _add_plot_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--output-dir', type=str, default='pseudospectrum',
                      help='Directory holding a finished run to render.')
  parser.add_argument('--plot-name', type=str, default='pseudoplot_default.html',
                      help='Base filename; _heatmap and _contours are derived from it.')
  parser.add_argument('--levels', type=float, nargs='+', default=None,
                      help='Explicit contour levels. Overrides --nlevels and '
                           '--min-level, which may not be given alongside it.')
  parser.add_argument('--nlevels', type=int, default=None,
                      help='Number of contour levels (default 16).')
  parser.add_argument('--min-level', type=float, default=None,
                      help='Minimum contour level (default 1e-7).')
  parser.add_argument('--plot-mesh', type=int, default=400,
                      help='Mesh resolution the heatmap interpolates onto. Contours '
                           'are drawn from the samples themselves and ignore this.')
  parser.add_argument('--plot-inline-js', action='store_true',
                      help='Embed plotly.js rather than linking the CDN, so the page '
                           'renders on a machine without network access.')


def build_parser() -> argparse.ArgumentParser:
  """Build the `nonmodal` argument parser."""
  parser = argparse.ArgumentParser(
    prog='nonmodal',
    description='Nonmodal (pseudospectral) stability diagnostics.')
  sub = parser.add_subparsers(dest='command', required=True)
  _add_run_arguments(sub.add_parser('run', help='Sample a pseudospectrum.'))
  _add_plot_arguments(sub.add_parser('plot', help='Render a finished run.'))
  return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse command line arguments."""
  return build_parser().parse_args(argv)


def _lattice_shape(args: argparse.Namespace) -> tuple[int, int]:
  """Resolve the lattice shape from a budget or an explicit nx/ny."""
  if (args.grid_nx is None) != (args.grid_ny is None):
    raise ValueError('give both --grid-nx and --grid-ny, or neither')
  if args.grid_nx is not None:
    return int(args.grid_nx), int(args.grid_ny)
  if args.grid_points < 1:
    raise ValueError('grid-points must be >= 1')
  return near_square(args.grid_points)


def _point_source(args: argparse.Namespace) -> PointSource:
  """Choose where to sample from.

  Explicit bounds win. Omitting all four infers the region from the spectrum
  once it has been computed. Giving only some is an error: a half-specified
  region is far more likely to be a mistake than a request to infer the rest.
  """
  given = {f'--{name.replace("_", "-")}': getattr(args, name) for name in BOUND_FLAGS}
  missing = [name for name, value in given.items() if value is None]

  if args.grid_npy:
    # A supplied point set fixes the region and the count, so anything that
    # would shape a lattice is inert.
    inert = [name for name, value in given.items() if value is not None]
    if args.grid_nx is not None or args.grid_ny is not None:
      inert.append('--grid-nx/--grid-ny')
    if inert:
      raise ValueError(
        f'--grid-npy supplies the sample points, so {", ".join(inert)} '
        'would have no effect')
    return FileSource(args.grid_npy)

  nx, ny = _lattice_shape(args)

  if len(missing) == len(given):
    return SpectrumSource(nx, ny, pad=args.bounds_pad)
  if missing:
    raise ValueError(
      'give all four real/imag bounds, or none of them to infer the region '
      f'from the spectrum; missing {", ".join(missing)}')

  bounds = Bounds(args.real_min, args.real_max, args.imag_min, args.imag_max)
  return RectangularSource(bounds, nx, ny)


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
  """Resolve parsed arguments into a `RunConfig`."""
  source = _point_source(args)

  if args.refine_rounds > 0 and not isinstance(source, RectangularSource | SpectrumSource):
    raise ValueError(
      '--refine-rounds needs a rectangular region to grow into; it cannot '
      'refine the fixed point set given by --grid-npy')

  return RunConfig(
    source=source,
    jacobian=args.jacobian,
    massmat=args.massmat,
    cache_dir=args.cache_dir,
    output_dir=args.output_dir,
    nprocs=args.nprocs,
    refine_rounds=args.refine_rounds,
    refine_seed_fraction=args.refine_seed_fraction,
    force_full_plane=args.no_half_plane,
    timestep=DEFAULT_TIMESTEP if args.timestep is None else args.timestep,
    n_eigvecs=DEFAULT_N_EIGVECS if args.n_eigvecs is None else args.n_eigvecs,
    run_tag=args.run_tag,
    case_tag=args.case_tag)


def plot_config_from_args(args: argparse.Namespace) -> PlotConfig:
  """Resolve parsed arguments into a `PlotConfig`."""
  if args.levels is not None:
    clashing = [
      flag for flag, value in (('--nlevels', args.nlevels),
                               ('--min-level', args.min_level))
      if value is not None
    ]
    if clashing:
      raise ValueError(
        f'--levels sets the contour levels directly, so {", ".join(clashing)} '
        'would have no effect')

  return PlotConfig(
    output_dir=args.output_dir,
    plot_name=args.plot_name,
    nlevels=DEFAULT_NLEVELS if args.nlevels is None else args.nlevels,
    min_level=DEFAULT_MIN_LEVEL if args.min_level is None else args.min_level,
    mesh=args.plot_mesh,
    inline_js=args.plot_inline_js,
    levels=tuple(args.levels) if args.levels else ())


def main(argv: list[str] | None = None) -> None:
  """CLI entry point."""
  args = parse_args(argv)
  if args.command == 'plot':
    plot_run(plot_config_from_args(args))
  else:
    run_pipeline(run_config_from_args(args))
