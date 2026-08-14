"""Reading simulation constants back from the input deck beside an export."""

import numpy as np
import pytest
from scipy import sparse

from nonmodal.adapters import (
  INPUT_DECK_NAME,
  find_input_deck,
  read_adiabatic_index,
  read_namelist_value,
)
from nonmodal.fields import FIELD_BLOCK_COUNT, TEMPERATURE_BLOCK_ID
from nonmodal.operator import mass_blocks

DECK = """&runtime_options
  ppn=1
/

&xmhd_options
  linear=T
  order=3
  chi=2943.4,
  gamma=1.67
  den_scale = 5.196374E16
  dt=1.E-7
/
"""


def _write(dirpath, text: str = DECK) -> str:
  path = dirpath / INPUT_DECK_NAME
  path.write_text(text)
  return str(path)


def test_gamma_is_read_from_the_simulation_namelist(tmp_path) -> None:
  assert read_adiabatic_index(_write(tmp_path)) == pytest.approx(1.67)


def test_a_value_is_scoped_to_its_namelist_group(tmp_path) -> None:
  """`ppn` lives in another group, so looking for it in xmhd_options must fail."""
  assert read_namelist_value(DECK, 'runtime_options', 'ppn') == '1'
  with pytest.raises(ValueError, match='no ppn'):
    read_namelist_value(DECK, 'xmhd_options', 'ppn')


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


def test_missing_group_and_missing_key_are_distinguished(tmp_path) -> None:
  with pytest.raises(ValueError, match='no &xmhd_options group'):
    read_namelist_value('&other\n/\n', 'xmhd_options', 'gamma')
  with pytest.raises(ValueError, match='no gamma'):
    read_namelist_value('&xmhd_options\n  dt=1.E-7\n/\n', 'xmhd_options', 'gamma')


def test_deck_is_found_beside_the_export(tmp_path) -> None:
  _write(tmp_path)
  assert find_input_deck(str(tmp_path / 'lin_ops.h5')) == str(tmp_path / INPUT_DECK_NAME)


def test_deck_is_found_in_a_parent_of_the_run_directory(tmp_path) -> None:
  """Run directories sit under a case root that holds the canonical deck."""
  _write(tmp_path)
  run = tmp_path / 'margin_finding'
  run.mkdir()
  assert find_input_deck(str(run / 'lin_ops.h5')) == str(tmp_path / INPUT_DECK_NAME)


def test_a_missing_deck_raises_rather_than_guessing(tmp_path) -> None:
  """A wrong gamma silently produces a wrong operator, so absence must be loud."""
  with pytest.raises(FileNotFoundError, match='define the operator'):
    find_input_deck(str(tmp_path / 'lin_ops.h5'), parents=0)


def test_only_the_temperature_block_is_scaled() -> None:
  scalar = sparse.csr_array(np.array([[2.0, 0.0], [0.0, 4.0]]))
  gamma = 1.67
  blocks = mass_blocks(scalar, gamma)

  assert len(blocks) == FIELD_BLOCK_COUNT
  for i, block in enumerate(blocks):
    expected = scalar.toarray() / (gamma - 1.0) if i == TEMPERATURE_BLOCK_ID \
      else scalar.toarray()
    np.testing.assert_allclose(block.toarray(), expected)


def test_the_temperature_block_matches_the_solver_factor() -> None:
  """OFT assembles the T mass term as basis*T*int_factor/(gamma-1)."""
  scalar = sparse.csr_array(np.eye(3))
  gamma = 1.67
  t_block = mass_blocks(scalar, gamma)[TEMPERATURE_BLOCK_ID].toarray()

  np.testing.assert_allclose(t_block, np.eye(3) / 0.67, rtol=1e-12)
  # Larger than the unscaled block, since gamma - 1 < 1 for a real plasma.
  assert t_block[0, 0] > 1.0
