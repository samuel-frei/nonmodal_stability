"""Argument parsing, config resolution, and validation."""

import pytest

from nonmodal.cli import parse_args, plot_config_from_args, run_config_from_args
from nonmodal.config import PlotConfig, RunConfig
from nonmodal.sampling import FileSource, RectangularSource, SpectrumSource

BOUNDS = ['--real-min', '-1', '--real-max', '1', '--imag-min', '-2', '--imag-max', '2']


def _run(*extra: str):
  return run_config_from_args(parse_args(['run', *BOUNDS, *extra]))


def test_subcommand_is_required() -> None:
  with pytest.raises(SystemExit):
    parse_args([])


def test_run_defaults_are_fully_resolved() -> None:
  config = _run()
  assert isinstance(config, RunConfig)
  assert config.jacobian == './lin_ops.h5'
  assert config.massmat == './mass_mat.h5'
  assert config.cache_dir == '.'
  assert config.refine_rounds == 0  # refinement is opt-in
  assert config.force_full_plane is False
  # Nothing on a resolved config is None.
  assert all(getattr(config, f) is not None for f in config.__dataclass_fields__)


def test_bounds_become_a_rectangular_source() -> None:
  source = _run('--grid-points', '64').source
  assert isinstance(source, RectangularSource)
  assert source.n_points == 64
  assert source.bounds.real_min == -1.0
  assert source.bounds.imag_max == 2.0


def test_operator_tunables_reach_the_config() -> None:
  config = _run('--timestep', '2.5e-6', '--n-eigvecs', '12',
                '--refine-rounds', '3', '--refine-seed-fraction', '0.2')
  assert config.timestep == pytest.approx(2.5e-6)
  assert config.n_eigvecs == 12
  assert config.refine_rounds == 3
  assert config.refine_seed_fraction == pytest.approx(0.2)


@pytest.mark.parametrize(
  ('flags', 'message'),
  [
    (['--refine-rounds', '4'], 'cannot refine the fixed point set'),
    (['--real-min', '-1', '--real-max', '1',
      '--imag-min', '-1', '--imag-max', '1'], 'would have no effect'),
    (['--grid-nx', '8', '--grid-ny', '8'], 'would have no effect'),
  ],
)
def test_flags_inert_under_grid_npy_are_rejected(
  flags: list[str], message: str
) -> None:
  """A flag that cannot take effect must fail loudly, not be ignored."""
  args = parse_args(['run', '--grid-npy', 'g.npy', *flags])
  with pytest.raises(ValueError, match=message):
    run_config_from_args(args)


def test_grid_npy_becomes_a_file_source_without_bounds() -> None:
  config = run_config_from_args(parse_args(['run', '--grid-npy', 'grid.npy']))
  assert isinstance(config.source, FileSource)
  assert config.source.path == 'grid.npy'


def test_omitting_all_bounds_infers_from_the_spectrum() -> None:
  config = run_config_from_args(parse_args(['run']))
  assert isinstance(config.source, SpectrumSource)
  assert config.source.pad == pytest.approx(0.3)


def test_bounds_pad_is_tunable() -> None:
  config = run_config_from_args(parse_args(['run', '--bounds-pad', '0.75']))
  assert isinstance(config.source, SpectrumSource)
  assert config.source.pad == pytest.approx(0.75)


def test_partial_bounds_are_rejected_and_named() -> None:
  """Half a region is a mistake, not a request to infer the other half."""
  args = parse_args(['run', '--real-min', '-1', '--real-max', '1'])
  with pytest.raises(ValueError, match='--imag-min, --imag-max'):
    run_config_from_args(args)


def test_grid_shape_can_be_given_explicitly() -> None:
  source = _run('--grid-nx', '32', '--grid-ny', '8').source
  assert (source.nx, source.ny) == (32, 8)
  assert source.n_points == 256


def test_half_a_grid_shape_is_rejected() -> None:
  with pytest.raises(ValueError, match='both --grid-nx and --grid-ny'):
    _run('--grid-nx', '32')


def test_grid_points_becomes_a_near_square_shape() -> None:
  source = _run('--grid-points', '400').source
  assert (source.nx, source.ny) == (20, 20)


def test_inverted_bounds_rejected() -> None:
  args = parse_args(['run', '--real-min', '1', '--real-max', '-1',
                     '--imag-min', '-2', '--imag-max', '2'])
  with pytest.raises(ValueError, match='strict bounds'):
    run_config_from_args(args)


@pytest.mark.parametrize(
  ('flags', 'message'),
  [
    (['--nprocs', '0'], 'nprocs must be >= 1'),
    (['--grid-points', '0'], 'grid-points must be >= 1'),
    (['--refine-rounds', '-1'], 'refine-rounds must be >= 0'),
  ],
)
def test_rejects_out_of_range_values(flags: list[str], message: str) -> None:
  with pytest.raises(ValueError, match=message):
    _run(*flags)


def test_no_half_plane_forces_full_sampling() -> None:
  assert _run('--no-half-plane').force_full_plane is True


def test_plot_config_defaults() -> None:
  config = plot_config_from_args(parse_args(['plot']))
  assert isinstance(config, PlotConfig)
  assert config.output_dir == 'pseudospectrum'
  assert config.mesh == 400
  assert config.inline_js is False


def test_plot_config_flags() -> None:
  config = plot_config_from_args(parse_args(
    ['plot', '--output-dir', 'out', '--plot-mesh', '64', '--plot-inline-js']))
  assert config.output_dir == 'out'
  assert config.mesh == 64
  assert config.inline_js is True


@pytest.mark.parametrize(
  ('flags', 'message'),
  [
    (['--nlevels', '0'], 'nlevels must be >= 1'),
    (['--min-level', '0'], 'min-level must be positive'),
    (['--plot-mesh', '1'], 'plot-mesh must be >= 2'),
  ],
)
def test_plot_validation(flags: list[str], message: str) -> None:
  with pytest.raises(ValueError, match=message):
    plot_config_from_args(parse_args(['plot', *flags]))


def test_explicit_levels_reach_the_config() -> None:
  config = plot_config_from_args(
    parse_args(['plot', '--levels', '1e-6', '1e-4', '1e-2']))
  assert config.levels == (1e-6, 1e-4, 1e-2)


@pytest.mark.parametrize('clashing', [['--nlevels', '8'], ['--min-level', '1e-5']])
def test_levels_conflicts_with_the_flags_it_overrides(clashing: list[str]) -> None:
  args = parse_args(['plot', '--levels', '1e-6', '1e-4', *clashing])
  with pytest.raises(ValueError, match='would have no effect'):
    plot_config_from_args(args)
