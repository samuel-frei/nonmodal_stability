"""Reading simulation constants back from the input deck beside an export."""

import numpy as np
import pytest
from scipy import sparse

from nonmodal.adapters import (
  MU0,
  PROTON_MASS,
  find_input_deck,
  read_adiabatic_index,
  read_namelist_value,
  read_physical_scales,
  read_resistivity,
  read_timestep,
  simulation_namelist,
)
from nonmodal.fields import FIELD_BLOCK_COUNT, PSI_BLOCK_ID, TEMPERATURE_BLOCK_ID
from nonmodal.operator import mass_blocks, psi_mass_scale

#: (deck filename, simulation group) for the upstream driver and the older one.
GENERATIONS = [('oft.in', 'harris_sheet_options'), ('oft_surf.in', 'xmhd_options')]

BODY = """
  linear=T
  order=3
  chi=2943.4,
  eta=9.3317868E-4
  gamma=1.67
  den_scale = 5.196374E16
  dt=1.E-7
/
"""


def _deck(group: str = 'harris_sheet_options') -> str:
  return f'&runtime_options\n  ppn=1\n/\n\n&{group}{BODY}'


DECK = _deck()


def _write(dirpath, text: str = DECK, name: str = 'oft.in') -> str:
  path = dirpath / name
  path.write_text(text)
  return str(path)


@pytest.mark.parametrize(('name', 'group'), GENERATIONS)
def test_both_deck_generations_are_read(tmp_path, name: str, group: str) -> None:
  path = _write(tmp_path, _deck(group), name)
  assert find_input_deck(str(tmp_path / 'lin_ops.h5')) == path
  assert read_adiabatic_index(path) == pytest.approx(1.67)
  assert read_resistivity(path) == pytest.approx(9.3317868e-4)


def test_a_value_is_scoped_to_its_namelist_group() -> None:
  """`ppn` lives in another group, so looking for it in the simulation group must fail."""
  assert read_namelist_value(DECK, 'runtime_options', 'ppn') == '1'
  with pytest.raises(ValueError, match='no ppn'):
    read_namelist_value(DECK, 'harris_sheet_options', 'ppn')


def test_fortran_d_exponents_are_accepted(tmp_path) -> None:
  assert read_adiabatic_index(
    _write(tmp_path, DECK.replace('gamma=1.67', 'gamma=1.6667d0'))
  ) == pytest.approx(1.6667)


def test_a_trailing_comma_does_not_bleed_into_the_value(tmp_path) -> None:
  assert read_adiabatic_index(
    _write(tmp_path, DECK.replace('gamma=1.67', 'gamma=1.4,'))) == pytest.approx(1.4)


@pytest.mark.parametrize('bad', ['gamma=1.0', 'gamma=0.5'])
def test_gamma_at_or_below_one_is_rejected(tmp_path, bad: str) -> None:
  """1/(gamma-1) is singular at 1 and negative below it."""
  with pytest.raises(ValueError, match='gamma must exceed 1'):
    read_adiabatic_index(_write(tmp_path, DECK.replace('gamma=1.67', bad)))


@pytest.mark.parametrize('bad', ['eta=0.0', 'eta=-1.E-3'])
def test_non_positive_eta_is_rejected(tmp_path, bad: str) -> None:
  with pytest.raises(ValueError, match='eta must be positive'):
    read_resistivity(_write(tmp_path, DECK.replace('eta=9.3317868E-4', bad)))


def test_missing_group_and_missing_key_are_distinguished() -> None:
  with pytest.raises(ValueError, match='no &xmhd_options group'):
    read_namelist_value('&other\n/\n', 'xmhd_options', 'gamma')
  with pytest.raises(ValueError, match='no gamma'):
    read_namelist_value('&xmhd_options\n  dt=1.E-7\n/\n', 'xmhd_options', 'gamma')


def test_a_deck_without_a_simulation_group_is_rejected() -> None:
  with pytest.raises(ValueError, match='no simulation group'):
    simulation_namelist('&runtime_options\n  ppn=1\n/\n')


def test_a_deck_with_both_simulation_groups_is_rejected() -> None:
  """The driver reads one group and ignores the other, so which one is ambiguous."""
  with pytest.raises(ValueError, match='both &harris_sheet_options and &xmhd_options'):
    simulation_namelist(_deck('harris_sheet_options') + _deck('xmhd_options'))


def test_deck_is_found_in_a_parent_of_the_run_directory(tmp_path) -> None:
  """Run directories sit under a case root that holds the canonical deck."""
  path = _write(tmp_path)
  run = tmp_path / 'margin_finding'
  run.mkdir()
  assert find_input_deck(str(run / 'lin_ops.h5')) == path


def test_the_nearest_deck_wins_whatever_its_name(tmp_path) -> None:
  _write(tmp_path, name='oft.in')
  run = tmp_path / 'run'
  run.mkdir()
  beside = _write(run, _deck('xmhd_options'), name='oft_surf.in')
  assert find_input_deck(str(run / 'lin_ops.h5')) == beside


def test_both_deck_names_at_one_level_are_rejected(tmp_path) -> None:
  _write(tmp_path, name='oft.in')
  _write(tmp_path, _deck('xmhd_options'), name='oft_surf.in')
  with pytest.raises(ValueError, match='both oft.in and oft_surf.in'):
    find_input_deck(str(tmp_path / 'lin_ops.h5'))


def test_a_missing_deck_raises_rather_than_guessing(tmp_path) -> None:
  """A wrong gamma silently produces a wrong operator, so absence must be loud."""
  with pytest.raises(FileNotFoundError, match='define the operator'):
    find_input_deck(str(tmp_path / 'lin_ops.h5'), parents=0)


def test_only_temperature_and_psi_blocks_are_scaled() -> None:
  scalar = sparse.csr_array(np.array([[2.0, 0.0], [0.0, 4.0]]))
  gamma, psi_scale = 1.67, 1.0 / 9.3317868e-4
  blocks = mass_blocks(scalar, gamma, psi_scale)

  assert len(blocks) == FIELD_BLOCK_COUNT
  for i, block in enumerate(blocks):
    factor = {TEMPERATURE_BLOCK_ID: 1.0 / (gamma - 1.0), PSI_BLOCK_ID: psi_scale}.get(i, 1.0)
    np.testing.assert_allclose(block.toarray(), scalar.toarray() * factor)


def test_the_temperature_block_matches_the_solver_factor() -> None:
  """OFT assembles the T mass term as basis*T*int_factor/(gamma-1)."""
  scalar = sparse.csr_array(np.eye(3))
  gamma = 1.67
  t_block = mass_blocks(scalar, gamma, 1.0)[TEMPERATURE_BLOCK_ID].toarray()

  np.testing.assert_allclose(t_block, np.eye(3) / 0.67, rtol=1e-12)
  # Larger than the unscaled block, since gamma - 1 < 1 for a real plasma.
  assert t_block[0, 0] > 1.0


ETA = 9.3317868e-4


def _local_export(psi_row_scale: float) -> tuple[sparse.csr_array, np.ndarray, sparse.csr_array]:
  """A local Jacobian shaped like OFT's: mass times row scale plus a small dt term.

  The first and last node are boundary rows, which OFT replaces with identity.
  """
  n = 6
  rng = np.random.default_rng(0)
  lap = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).toarray()
  mass = sparse.csr_array(np.eye(n) * 4.0 + sparse.diags([1.0, 1.0], [-1, 1], shape=(n, n)))
  row_scale = [1.0] * FIELD_BLOCK_COUNT
  row_scale[TEMPERATURE_BLOCK_ID] = 1.0 / 0.67
  row_scale[PSI_BLOCK_ID] = psi_row_scale
  bc = np.zeros(n * FIELD_BLOCK_COUNT, dtype=bool)
  blocks = []
  for b, scale in enumerate(row_scale):
    block = scale * (mass.toarray() + 1e-2 * lap) + 1e-3 * rng.standard_normal((n, n))
    for edge in (0, n - 1):
      block[edge] = 0.0
      block[edge, edge] = 1.0
      bc[b * n + edge] = True
    blocks.append(block)
  return sparse.csr_array(sparse.block_diag(blocks)), bc, mass


@pytest.mark.parametrize('row_scale', [1.0, 1.0 / ETA])
def test_psi_mass_scale_is_read_off_the_jacobian(row_scale: float) -> None:
  """Old OFT leaves psi rows unscaled; upstream divides them by eta."""
  jac, bc, mass = _local_export(row_scale)
  assert psi_mass_scale(jac, bc, mass, ETA) == pytest.approx(row_scale)


def test_a_psi_ratio_near_neither_candidate_raises() -> None:
  """Picking the nearer of two wrong answers would silently build a wrong operator."""
  jac, bc, mass = _local_export(30.0)
  with pytest.raises(ValueError, match='near neither'):
    psi_mass_scale(jac, bc, mass, ETA)


def test_a_deck_eta_that_disagrees_with_the_export_raises() -> None:
  """Scaled rows with a different deck eta land near neither candidate."""
  jac, bc, mass = _local_export(1.0 / ETA)
  with pytest.raises(ValueError, match='near neither'):
    psi_mass_scale(jac, bc, mass, ETA * 10)


def _scales_deck(group: str, eta: float, nu: float) -> str:
  return f"""&{group}
  n0=5.196374E16
  psi0=0.002045692328575E0,
  eta={eta!r}
  nu={nu!r}
  gamma=1.67
  dt=1.E-7
  lam_b = 0.6075E0
/
"""


def test_both_unit_conventions_give_the_same_physical_scales(tmp_path) -> None:
  """The old driver's Ohm m and Pa s are the new driver's m^2/s, converted."""
  diffusivity, viscosity = 742.6, 7424.2
  mass_density = PROTON_MASS * 5.196374e16
  new = _write(tmp_path, _scales_deck('harris_sheet_options', diffusivity, viscosity), 'oft.in')
  old_dir = tmp_path / 'old'
  old_dir.mkdir()
  old = _write(old_dir, _scales_deck('xmhd_options', diffusivity * MU0, viscosity * mass_density),
               'oft_surf.in')

  a, b = read_physical_scales(new), read_physical_scales(old)
  for field in ('magnetic_diffusivity_m2_s', 'kinematic_viscosity_m2_s', 'alfven_time_s',
                'lundquist_number', 'magnetic_prandtl_number'):
    assert getattr(a, field) == pytest.approx(getattr(b, field), rel=1e-12)
  assert a.unit_convention != b.unit_convention


def test_physical_scales_match_the_closed_form(tmp_path) -> None:
  scales = read_physical_scales(
    _write(tmp_path, _scales_deck('harris_sheet_options', 742.6, 7424.2)))
  v_a = 0.002045692328575 / np.sqrt(MU0 * PROTON_MASS * 5.196374e16)

  assert scales.alfven_speed_m_s == pytest.approx(v_a, rel=1e-12)
  assert scales.alfven_time_s == pytest.approx(0.6075 / v_a, rel=1e-12)
  assert scales.lundquist_number == pytest.approx(0.6075 * v_a / 742.6, rel=1e-12)
  assert scales.magnetic_prandtl_number == pytest.approx(7424.2 / 742.6, rel=1e-12)
  # The 20x6z case: tau_A of a few microseconds, S ~ 160, Pm ~ 10.
  assert 3.0e-6 < scales.alfven_time_s < 3.2e-6
  assert 150 < scales.lundquist_number < 170


def test_timestep_is_read_from_the_deck(tmp_path) -> None:
  assert read_timestep(_write(tmp_path)) == pytest.approx(1e-7)
