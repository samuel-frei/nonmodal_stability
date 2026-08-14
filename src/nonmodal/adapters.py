"""Read simulation parameters from the OpenFUSIONToolkit input deck.

The exported matrices do not carry the physical constants that define the
operator, so they are read back from the `oft_surf.in` namelist that ships
beside them. Only the values the reduced operator needs are parsed.

* `find_input_deck` -- locate `oft_surf.in` at or above a matrix export.
* `read_namelist_value` -- one scalar out of one Fortran namelist group.
* `read_adiabatic_index` -- `gamma`, which scales the temperature mass block.
"""

import os
import re

INPUT_DECK_NAME = 'oft_surf.in'
SIMULATION_NAMELIST = 'xmhd_options'
#: Levels above the export to search; run directories sit under a case root.
SEARCH_PARENTS = 3


def find_input_deck(start_path: str, parents: int = SEARCH_PARENTS) -> str:
  """Locate `oft_surf.in` beside `start_path`, then in its parent directories."""
  directory = os.path.dirname(os.path.abspath(start_path))
  for _ in range(parents + 1):
    candidate = os.path.join(directory, INPUT_DECK_NAME)
    if os.path.isfile(candidate):
      return candidate
    parent = os.path.dirname(directory)
    if parent == directory:
      break
    directory = parent
  raise FileNotFoundError(
    f'no {INPUT_DECK_NAME} found beside {start_path} or within {parents} '
    f'parent directories; it carries the constants that define the operator')


def read_namelist_value(text: str, namelist: str, key: str) -> str:
  """Return the raw assigned text for `key` inside `&namelist`."""
  # A namelist group runs from &name to a lone terminating slash.
  group = re.search(
    rf'&{re.escape(namelist)}\b(.*?)^\s*/\s*$',
    text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
  if group is None:
    raise ValueError(f'no &{namelist} group in the input deck')
  assigned = re.search(
    rf'^\s*{re.escape(key)}\s*=\s*([^,\s!]+)', group.group(1),
    re.IGNORECASE | re.MULTILINE)
  if assigned is None:
    raise ValueError(f'no {key} in &{namelist}')
  return assigned.group(1)


def read_adiabatic_index(deck_path: str) -> float:
  """Read `gamma` from an input deck, as a float."""
  with open(deck_path, encoding='utf-8', errors='replace') as f:
    raw = read_namelist_value(f.read(), SIMULATION_NAMELIST, 'gamma')
  # Fortran writes exponents with d as readily as e.
  gamma = float(raw.lower().replace('d', 'e'))
  if gamma <= 1.0:
    raise ValueError(
      f'gamma must exceed 1 for the 1/(gamma-1) temperature mass term, got {gamma}')
  return gamma
