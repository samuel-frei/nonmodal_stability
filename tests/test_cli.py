"""Argument parsing and pipeline input validation."""

import pytest

from nonmodal.cli import parse_args
from nonmodal.pipeline import validate_and_normalize_args

BOUNDS = ['--real-min', '-1', '--real-max', '1', '--imag-min', '-2', '--imag-max', '2']


def test_defaults_include_configurable_paths() -> None:
  args = parse_args(BOUNDS)
  assert args.jacobian == './lin_ops.h5'
  assert args.massmat == './mass_mat.h5'
  assert args.cache_dir == '.'


def test_validate_returns_bounds_and_normalizes_plot_name() -> None:
  args = parse_args([*BOUNDS, '--plot-name', 'out'])
  assert validate_and_normalize_args(args) == (-1.0, 1.0, -2.0, 2.0)
  assert args.plot_name == 'out.html'


def test_bounds_required_without_grid_npy() -> None:
  args = parse_args([])
  with pytest.raises(ValueError, match='unless --grid-npy is used'):
    validate_and_normalize_args(args)


def test_grid_npy_makes_bounds_optional() -> None:
  args = parse_args(['--grid-npy', 'grid.npy'])
  assert validate_and_normalize_args(args) == (None, None, None, None)


def test_inverted_bounds_rejected() -> None:
  args = parse_args(['--real-min', '1', '--real-max', '-1',
                     '--imag-min', '-2', '--imag-max', '2'])
  with pytest.raises(ValueError, match='strict bounds'):
    validate_and_normalize_args(args)


@pytest.mark.parametrize(
  ('flags', 'message'),
  [
    (['--nprocs', '0'], 'nprocs must be >= 1'),
    (['--grid-points', '0'], 'grid-points must be >= 1'),
    (['--nlevels', '0'], 'nlevels must be >= 1'),
    (['--min-level', '0'], 'min-level must be positive'),
  ],
)
def test_rejects_out_of_range_values(flags: list[str], message: str) -> None:
  args = parse_args([*BOUNDS, *flags])
  with pytest.raises(ValueError, match=message):
    validate_and_normalize_args(args)


def test_grid_shape_must_be_positive() -> None:
  args = parse_args(['--grid-npy', 'g.npy', '--grid-shape', '0', '4'])
  with pytest.raises(ValueError, match='grid-shape rows and cols must be >= 1'):
    validate_and_normalize_args(args)
