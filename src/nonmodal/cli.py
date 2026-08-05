"""Command-line interface.

Two subcommands, because computing and looking at the result have different
requirements: `run` needs the HDF5 matrices and a machine with cores, `plot`
needs only a finished output directory.

This is the only module that touches `argparse.Namespace`; everything below it
receives a resolved `RunConfig` or `PlotConfig`.
"""

import argparse

from .config import PlotConfig, RunConfig
from .pipeline import plot_run, run_pipeline
from .sampling import Bounds, FileSource, PointSource, RectangularSource


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--grid-points', type=int, default=128,
                      help='Total number of sigma_min evaluations to spend.')
  parser.add_argument('--nprocs', type=int, default=128, help='Worker process count.')
  parser.add_argument('--jacobian', type=str, default='./lin_ops.h5',
                      help='HDF5 file holding the /jacobian matrix.')
  parser.add_argument('--massmat', type=str, default='./mass_mat.h5',
                      help='HDF5 file holding the /massmat matrix.')
  parser.add_argument('--cache-dir', type=str, default='.',
                      help='Directory for the reduced-operator, eigenvalue and Schur '
                           'caches. Caches are reused by filename alone and are NOT '
                           'invalidated when inputs change; delete them by hand after '
                           'changing inputs.')
  parser.add_argument('--grid-npy', type=str, default='',
                      help='Optional .npy file of flat complex points to sample, '
                           'instead of a rectangular region.')
  parser.add_argument('--real-min', type=float, default=None,
                      help='Sampled real-axis minimum (required unless --grid-npy).')
  parser.add_argument('--real-max', type=float, default=None,
                      help='Sampled real-axis maximum (required unless --grid-npy).')
  parser.add_argument('--imag-min', type=float, default=None,
                      help='Sampled imaginary-axis minimum (required unless --grid-npy).')
  parser.add_argument('--imag-max', type=float, default=None,
                      help='Sampled imaginary-axis maximum (required unless --grid-npy).')
  parser.add_argument('--refine-rounds', type=int, default=0,
                      help='Adaptive refinement rounds (0 disables it, the default). '
                           'Refinement measurably beats uniform sampling on strongly '
                           'non-normal operators and is roughly a wash otherwise.')
  parser.add_argument('--no-half-plane', action='store_true',
                      help='Sample the full plane even for a real operator, instead of '
                           'halving the work via conjugate symmetry.')
  parser.add_argument('--run-tag', type=str, default='',
                      help='Batch-level run identifier for metadata tracking.')
  parser.add_argument('--case-tag', type=str, default='',
                      help='Case identifier for metadata tracking.')
  parser.add_argument('--output-dir', type=str, default='pseudospectrum',
                      help='Output directory for samples and metadata.')


def _add_plot_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--output-dir', type=str, default='pseudospectrum',
                      help='Directory holding a finished run to render.')
  parser.add_argument('--plot-name', type=str, default='pseudoplot_default.html',
                      help='Base filename; _heatmap and _contours are derived from it.')
  parser.add_argument('--nlevels', type=int, default=16,
                      help='Number of contour levels.')
  parser.add_argument('--min-level', type=float, default=1e-7,
                      help='Minimum contour level.')
  parser.add_argument('--plot-mesh', type=int, default=400,
                      help='Mesh resolution the heatmap interpolates onto. Contours are '
                           'drawn from the samples themselves and ignore this.')
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


def _point_source(args: argparse.Namespace) -> PointSource:
  """Choose where to sample from, validating the bounds if given."""
  if args.grid_npy:
    return FileSource(args.grid_npy)

  missing = [
    name for name, value in (
      ('--real-min', args.real_min), ('--real-max', args.real_max),
      ('--imag-min', args.imag_min), ('--imag-max', args.imag_max))
    if value is None
  ]
  if missing:
    raise ValueError(
      'real/imag bounds must be provided unless --grid-npy is used; '
      f'missing {", ".join(missing)}')
  if args.grid_points < 1:
    raise ValueError('grid-points must be >= 1')

  bounds = Bounds(args.real_min, args.real_max, args.imag_min, args.imag_max)
  return RectangularSource(bounds, args.grid_points)


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
  """Resolve parsed arguments into a `RunConfig`."""
  return RunConfig(
    source=_point_source(args),
    jacobian=args.jacobian,
    massmat=args.massmat,
    cache_dir=args.cache_dir,
    output_dir=args.output_dir,
    nprocs=args.nprocs,
    refine_rounds=args.refine_rounds,
    force_full_plane=args.no_half_plane,
    run_tag=args.run_tag,
    case_tag=args.case_tag)


def plot_config_from_args(args: argparse.Namespace) -> PlotConfig:
  """Resolve parsed arguments into a `PlotConfig`."""
  return PlotConfig(
    output_dir=args.output_dir,
    plot_name=args.plot_name,
    nlevels=args.nlevels,
    min_level=args.min_level,
    mesh=args.plot_mesh,
    inline_js=args.plot_inline_js)


def main(argv: list[str] | None = None) -> None:
  """CLI entry point."""
  args = parse_args(argv)
  if args.command == 'plot':
    plot_run(plot_config_from_args(args))
  else:
    run_pipeline(run_config_from_args(args))
