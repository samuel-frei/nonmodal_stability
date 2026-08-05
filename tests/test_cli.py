"""Argument parsing, config resolution, and validation."""

import pytest

from nonmodal.cli import parse_args, plot_config_from_args, run_config_from_args
from nonmodal.config import PlotConfig, RunConfig
from nonmodal.sampling import FileSource, SpectrumSource


def _run(*extra: str) -> RunConfig:
  return run_config_from_args(parse_args(['run', *extra]))


def test_subcommand_is_required() -> None:
  with pytest.raises(SystemExit):
    parse_args([])


def test_run_defaults_are_fully_resolved() -> None:
  config = _run()
  assert isinstance(config, RunConfig)
  assert config.jacobian == './lin_ops.h5'
  assert config.massmat == './mass_mat.h5'
  assert config.cache_dir == '.'
  assert config.refine_points == 0  # refinement is opt-in
  assert config.force_full_plane is False
  # Nothing on a resolved config is None.
  assert all(getattr(config, f) is not None for f in config.__dataclass_fields__)


def test_the_region_always_comes_from_the_spectrum() -> None:
  """There is no way to hand-pick a rectangle any more."""
  source = _run().source
  assert isinstance(source, SpectrumSource)
  assert source.pad == pytest.approx(0.3)
  assert (source.nx, source.ny) == (24, 24)


@pytest.mark.parametrize('flag', ['--real-min', '--real-max', '--imag-min',
                                  '--imag-max', '--grid-points'])
def test_removed_flags_are_gone(flag: str) -> None:
  with pytest.raises(SystemExit):
    parse_args(['run', flag, '1'])


def test_grid_dimensions_are_explicit() -> None:
  source = _run('--grid-nx', '32', '--grid-ny', '8').source
  assert (source.nx, source.ny) == (32, 8)
  assert source.n_points == 256


def test_bounds_pad_is_tunable() -> None:
  assert _run('--bounds-pad', '0.75').source.pad == pytest.approx(0.75)


def test_grid_npy_becomes_a_file_source() -> None:
  config = _run('--grid-npy', 'grid.npy')
  assert isinstance(config.source, FileSource)
  assert config.source.path == 'grid.npy'


def test_operator_tunables_reach_the_config() -> None:
  config = _run('--timestep', '2.5e-6', '--n-eigvecs', '12',
                '--refine-points', '500', '--refine-rounds', '3')
  assert config.timestep == pytest.approx(2.5e-6)
  assert config.n_eigvecs == 12
  assert config.refine_points == 500
  assert config.refine_rounds == 3


@pytest.mark.parametrize(
  ('flags', 'message'),
  [
    (['--nprocs', '0'], 'nprocs must be >= 1'),
    (['--grid-nx', '0'], 'grid-nx and grid-ny must be >= 1'),
    (['--refine-points', '-1'], 'refine-points must be >= 0'),
    (['--refine-points', '10', '--refine-rounds', '0'], 'refine-rounds must be >= 1'),
    (['--bounds-pad', '-0.1'], 'bounds-pad must be >= 0'),
  ],
)
def test_rejects_out_of_range_values(flags: list[str], message: str) -> None:
  with pytest.raises(ValueError, match=message):
    _run(*flags)


def test_no_half_plane_forces_full_sampling() -> None:
  assert _run('--no-half-plane').force_full_plane is True


@pytest.mark.parametrize(
  ('flags', 'message'),
  [
    (['--refine-points', '100'], 'cannot refine the fixed point set'),
    (['--grid-nx', '8'], 'would have no effect'),
    (['--bounds-pad', '0.9'], 'would have no effect'),
  ],
)
def test_flags_inert_under_grid_npy_are_rejected(
  flags: list[str], message: str
) -> None:
  """A flag that cannot take effect must fail loudly, not be ignored."""
  with pytest.raises(ValueError, match=message):
    _run('--grid-npy', 'g.npy', *flags)


def test_refine_rounds_without_points_is_rejected() -> None:
  """Rounds only divide up a budget; with no budget they do nothing."""
  with pytest.raises(ValueError, match='would have no effect'):
    _run('--refine-rounds', '8')


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
