"""Command-line interface."""

import argparse

from .pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse command line arguments for pseudospectrum generation."""
  parser = argparse.ArgumentParser(description='Compute reduced pseudospectrum and save outputs.')
  parser.add_argument('--grid-points', type=int, default=128,
                      help='Total number of grid points (minimum 128).')
  parser.add_argument('--nprocs', type=int, default=128, help='Worker process count.')
  parser.add_argument('--jacobian', type=str, default='./lin_ops.h5',
                      help='HDF5 file holding the /jacobian matrix.')
  parser.add_argument('--massmat', type=str, default='./mass_mat.h5',
                      help='HDF5 file holding the /massmat matrix.')
  parser.add_argument('--cache-dir', type=str, default='.',
                      help='Directory for the reduced-operator, eigenvalue and Schur caches. '
                           'Caches are reused by filename alone and are NOT invalidated when '
                           'inputs change; delete them by hand after changing inputs.')
  parser.add_argument('--grid-npy', type=str, default='',
                      help='Optional .npy file containing a flat complex grid to sample.')
  parser.add_argument('--grid-shape', type=int, nargs=2, metavar=('ROWS', 'COLS'),
                      help='Optional reshape for structured plots when using --grid-npy.')
  parser.add_argument('--real-min', type=float, default=None,
                      help='Sampled real-axis minimum (required unless --grid-npy is used).')
  parser.add_argument('--real-max', type=float, default=None,
                      help='Sampled real-axis maximum (required unless --grid-npy is used).')
  parser.add_argument('--imag-min', type=float, default=None,
                      help='Sampled imaginary-axis minimum (required unless --grid-npy is used).')
  parser.add_argument('--imag-max', type=float, default=None,
                      help='Sampled imaginary-axis maximum (required unless --grid-npy is used).')
  parser.add_argument('--nlevels', type=int, default=16,
                      help='Number of contour levels for interactive pseudospectrum plot.')
  parser.add_argument('--min-level', type=float, default=1e-7, help='Minimum contour level.')
  parser.add_argument('--plot-name', type=str, default='pseudoplot_default.html',
                      help='Interactive contour plot output filename.')
  parser.add_argument('--run-tag', type=str, default='',
                      help='Batch-level run identifier for metadata tracking.')
  parser.add_argument('--case-tag', type=str, default='',
                      help='Case identifier for metadata tracking.')
  parser.add_argument('--output-dir', type=str, default='pseudospectrum',
                      help='Output directory for arrays and metadata.')
  return parser.parse_args(argv)


def main() -> None:
  """CLI entry point."""
  run_pipeline(parse_args())
