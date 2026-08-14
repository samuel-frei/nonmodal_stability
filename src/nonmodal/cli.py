"""Command-line interface.

Three subcommands, because their requirements differ: `run` needs the HDF5
matrices and a machine with cores, `pseudomode` needs the operator but not the
cores, `plot` needs only a finished output directory. This is the only module
that touches `argparse.Namespace`. A flag that cannot take effect in the given
mode raises rather than being ignored.

* `build_parser` / `parse_args` -- the argparse surface.
* `run_config_from_args` / `pseudomode_config_from_args` /
  `plot_config_from_args` -- resolve a `Namespace` into a frozen config.
* `main` -- console entry point; dispatches to `pipeline`.
"""

import argparse

from .config import (
  DEFAULT_MIN_LEVEL,
  DEFAULT_MODE_DIR,
  DEFAULT_N_EIGVECS,
  DEFAULT_NLEVELS,
  DEFAULT_REFINE_ROUNDS,
  PlotConfig,
  PseudomodeConfig,
  RunConfig,
)
from .operator import DEFAULT_TIMESTEP
from .pipeline import plot_run, pseudomode_run, run_pipeline
from .pseudospectrum import DEFAULT_MODE_TOL
from .sampling import (
  DEFAULT_BOUNDS_PAD,
  DEFAULT_GRID_NX,
  DEFAULT_GRID_NY,
  FileSource,
  PointSource,
  SpectrumSource,
)


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
  grid = parser.add_argument_group('initial grid')
  grid.add_argument('--grid-nx', type=int, default=DEFAULT_GRID_NX,
                    help=f'Columns in the initial lattice (default '
                         f'{DEFAULT_GRID_NX}). Keep it coarse and spend the rest '
                         f'of the budget on --refine-points.')
  grid.add_argument('--grid-ny', type=int, default=DEFAULT_GRID_NY,
                    help=f'Rows in the initial lattice (default {DEFAULT_GRID_NY}).')
  grid.add_argument('--bounds-pad', type=float, default=DEFAULT_BOUNDS_PAD,
                    help='Padding around the spectrum, as a fraction of the '
                         'spectral span. The sampled region is always derived '
                         'from the spectrum.')
  grid.add_argument('--grid-npy', type=str, default='',
                    help='Optional .npy file of flat complex points to sample '
                         'instead of the spectrum-derived lattice.')

  refine = parser.add_argument_group('adaptive refinement')
  refine.add_argument('--refine-points', type=int, default=0,
                      help='Spend up to this many extra evaluations refining onto '
                           'features, on top of the initial grid (0 disables '
                           'refinement). A ceiling, not a guarantee: each round '
                           'inserts at most one point per triangle, so a large '
                           'request needs several --refine-rounds. This is the '
                           'intended way to reach resolution -- it measurably beats '
                           'sampling a finer uniform lattice on strongly non-normal '
                           'operators.')
  refine.add_argument('--refine-rounds', type=int, default=DEFAULT_REFINE_ROUNDS,
                      help=f'Rounds to spread --refine-points over (default '
                           f'{DEFAULT_REFINE_ROUNDS}). More rounds adapt more '
                           f'closely, at one worker-pool round trip each.')

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


def _complex_arg(text: str) -> complex:
  """Parse a complex literal such as `5e5-2.4e4j`."""
  try:
    return complex(text.replace(' ', ''))
  except ValueError:
    raise argparse.ArgumentTypeError(
      f'{text!r} is not a complex number; write it like 5e5-2.4e4j') from None


def _add_pseudomode_arguments(parser: argparse.ArgumentParser) -> None:
  parser.add_argument('--at', type=_complex_arg, action='append', default=None,
                      metavar='Z', required=True, dest='at',
                      help='Complex point to extract a mode at, e.g. 5e5-2.4e4j. '
                           'Repeat for several; they share one load of the '
                           'operator.')

  out = parser.add_argument_group('output')
  out.add_argument('--phases', type=int, default=1,
                   help='Restart files per mode (default 1). One uses the phase '
                        'carrying the most amplitude; more sweep the phase, so '
                        'the directory reads back as an animation.')
  out.add_argument('--mode-dir', type=str, default=DEFAULT_MODE_DIR,
                   help=f'Subdirectory of --output-dir for the restart files '
                        f'(default {DEFAULT_MODE_DIR}).')
  out.add_argument('--mode-tol', type=float, default=None,
                   help=f'ARPACK tolerance for the mode (default '
                        f'{DEFAULT_MODE_TOL:g}). Tighter than sampling uses, '
                        f'because the vector converges more slowly than the '
                        f'value.')

  op = parser.add_argument_group('operator')
  op.add_argument('--jacobian', type=str, default='./lin_ops.h5',
                  help='HDF5 file holding the /jacobian matrix. Needed for the '
                       'field-block reduction mapping even on a cache hit.')
  op.add_argument('--massmat', type=str, default='./mass_mat.h5',
                  help='HDF5 file holding the /massmat matrix.')
  op.add_argument('--timestep', type=float, default=None,
                  help='Timestep of the implicit solve that produced the Jacobian.')
  op.add_argument('--cache-dir', type=str, default='.',
                  help='Directory holding the operator and Schur caches. A run '
                       'whose Schur factor predates the vector cache has to redo '
                       'the factorisation, since Z cannot be recovered from T.')
  op.add_argument('--output-dir', type=str, default='pseudospectrum',
                  help='Directory the modes and their metadata are written to.')
  op.add_argument('--run-tag', type=str, default='',
                  help='Batch-level run identifier for metadata tracking.')
  op.add_argument('--case-tag', type=str, default='',
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
  _add_pseudomode_arguments(sub.add_parser(
    'pseudomode', help='Extract pseudomodes as restart files.'))
  return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse command line arguments."""
  return build_parser().parse_args(argv)


def _point_source(args: argparse.Namespace) -> PointSource:
  """Choose where to sample from: always the spectrum, or a supplied point set."""
  # The region is always the spectrum's; --grid-npy supplies a foreign set.
  if args.grid_npy:
    # That set fixes region and resolution, so the lattice flags are inert.
    inert = [
      flag for flag, given in (
        ('--grid-nx', args.grid_nx != DEFAULT_GRID_NX),
        ('--grid-ny', args.grid_ny != DEFAULT_GRID_NY),
        ('--bounds-pad', args.bounds_pad != DEFAULT_BOUNDS_PAD))
      if given
    ]
    if inert:
      raise ValueError(
        f'--grid-npy supplies the sample points, so {", ".join(inert)} '
        'would have no effect')
    return FileSource(args.grid_npy)

  return SpectrumSource(args.grid_nx, args.grid_ny, pad=args.bounds_pad)


def run_config_from_args(args: argparse.Namespace) -> RunConfig:
  """Resolve parsed arguments into a `RunConfig`."""
  source = _point_source(args)

  if args.refine_points > 0 and isinstance(source, FileSource):
    raise ValueError(
      '--refine-points needs a lattice to grow from; it cannot refine the '
      'fixed point set given by --grid-npy')
  if args.refine_points < 1 and args.refine_rounds != DEFAULT_REFINE_ROUNDS:
    raise ValueError(
      '--refine-rounds only divides up --refine-points, which is 0, so it '
      'would have no effect')

  return RunConfig(
    source=source,
    jacobian=args.jacobian,
    massmat=args.massmat,
    cache_dir=args.cache_dir,
    output_dir=args.output_dir,
    nprocs=args.nprocs,
    refine_points=args.refine_points,
    refine_rounds=args.refine_rounds,
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


def pseudomode_config_from_args(args: argparse.Namespace) -> PseudomodeConfig:
  """Resolve parsed arguments into a `PseudomodeConfig`."""
  return PseudomodeConfig(
    points=tuple(args.at) if args.at else (),
    jacobian=args.jacobian,
    massmat=args.massmat,
    cache_dir=args.cache_dir,
    output_dir=args.output_dir,
    mode_dir=args.mode_dir,
    phases=args.phases,
    tol=DEFAULT_MODE_TOL if args.mode_tol is None else args.mode_tol,
    timestep=DEFAULT_TIMESTEP if args.timestep is None else args.timestep,
    run_tag=args.run_tag,
    case_tag=args.case_tag)


def main(argv: list[str] | None = None) -> None:
  """CLI entry point."""
  args = parse_args(argv)
  if args.command == 'plot':
    plot_run(plot_config_from_args(args))
  elif args.command == 'pseudomode':
    pseudomode_run(pseudomode_config_from_args(args))
  else:
    run_pipeline(run_config_from_args(args))
