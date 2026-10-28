"""Read simulation parameters from the OpenFUSIONToolkit input deck.

The exported matrices do not carry the physical constants that define the
operator, so they are read back from the input deck that ships beside them.
Only the values the reduced operator needs are parsed.

Two generations of deck exist. The upstream Harris driver reads `oft.in` with
an `&harris_sheet_options` group; the older linear driver read `oft_surf.in`
with `&xmhd_options`. Both carry the same keys.

* `find_input_deck` -- locate the input deck at or above a matrix export.
* `read_namelist_value` -- one scalar out of one Fortran namelist group.
* `simulation_namelist` -- which simulation group a deck uses.
* `read_adiabatic_index` -- `gamma`, which scales the temperature mass block.
* `read_resistivity` -- `eta`, which can scale the psi mass block.
* `read_timestep` -- `dt`, which the exported Jacobian is linear in.
* `read_physical_scales` -- Alfven speed and time, Lundquist and magnetic
  Prandtl numbers, in whichever units the deck's driver used.
"""

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

#: Deck filenames, newest driver first.
INPUT_DECK_NAMES = ('oft.in', 'oft_surf.in')
#: Simulation namelist groups, newest driver first.
SIMULATION_NAMELISTS = ('harris_sheet_options', 'xmhd_options')
#: Levels above the export to search; run directories sit under a case root.
SEARCH_PARENTS = 3
#: OFT's own constants (`mhd_utils.F90`), so derived scales match the solver.
MU0 = math.pi * 4.0e-7
PROTON_MASS = 1.672621777e-27


def find_input_deck(start_path: str, parents: int = SEARCH_PARENTS) -> str:
  """Locate the input deck beside `start_path`, then in its parent directories.

  The nearest level holding a deck wins. A level holding both names is an
  error rather than a guess, since the wrong deck yields a wrong operator.
  """
  directory = os.path.dirname(os.path.abspath(start_path))
  for _ in range(parents + 1):
    found = [os.path.join(directory, name) for name in INPUT_DECK_NAMES
             if os.path.isfile(os.path.join(directory, name))]
    if len(found) > 1:
      raise ValueError(
        f'both {" and ".join(INPUT_DECK_NAMES)} exist in {directory}; '
        f'remove the one that did not produce the export')
    if found:
      return found[0]
    parent = os.path.dirname(directory)
    if parent == directory:
      break
    directory = parent
  raise FileNotFoundError(
    f'no {" or ".join(INPUT_DECK_NAMES)} found beside {start_path} or within '
    f'{parents} parent directories; it carries the constants that define the '
    f'operator')


def _group_body(text: str, namelist: str) -> str | None:
  """The body of `&namelist`, from its name to a lone terminating slash."""
  group = re.search(
    rf'&{re.escape(namelist)}\b(.*?)^\s*/\s*$',
    text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
  return None if group is None else group.group(1)


def read_namelist_value(text: str, namelist: str, key: str) -> str:
  """Return the raw assigned text for `key` inside `&namelist`."""
  body = _group_body(text, namelist)
  if body is None:
    raise ValueError(f'no &{namelist} group in the input deck')
  assigned = re.search(
    rf'^\s*{re.escape(key)}\s*=\s*([^,\s!]+)', body, re.IGNORECASE | re.MULTILINE)
  if assigned is None:
    raise ValueError(f'no {key} in &{namelist}')
  return assigned.group(1)


def simulation_namelist(text: str) -> str:
  """The simulation group this deck uses; exactly one must be present."""
  present = [name for name in SIMULATION_NAMELISTS if _group_body(text, name) is not None]
  if not present:
    raise ValueError(
      f'no simulation group in the input deck; expected one of '
      f'{", ".join("&" + name for name in SIMULATION_NAMELISTS)}')
  if len(present) > 1:
    raise ValueError(
      f'the input deck has both {" and ".join("&" + name for name in present)}; '
      f'the driver reads only one, so the other would be silently ignored')
  return present[0]


def _read_simulation_float(deck_path: str, key: str) -> float:
  """One numeric value from the deck's simulation group."""
  with open(deck_path, encoding='utf-8', errors='replace') as f:
    text = f.read()
  raw = read_namelist_value(text, simulation_namelist(text), key)
  # Fortran writes exponents with d as readily as e.
  return float(raw.lower().replace('d', 'e'))


def read_adiabatic_index(deck_path: str) -> float:
  """Read `gamma` from an input deck, as a float."""
  gamma = _read_simulation_float(deck_path, 'gamma')
  if gamma <= 1.0:
    raise ValueError(
      f'gamma must exceed 1 for the 1/(gamma-1) temperature mass term, got {gamma}')
  return gamma


def read_resistivity(deck_path: str) -> float:
  """Read the scalar `eta` from an input deck, as a float, in the deck's own units."""
  eta = _read_simulation_float(deck_path, 'eta')
  if eta <= 0.0:
    raise ValueError(f'eta must be positive for the 1/eta psi mass term, got {eta}')
  return eta


def read_timestep(deck_path: str) -> float:
  """Read `dt` [s] from an input deck."""
  dt = _read_simulation_float(deck_path, 'dt')
  if dt <= 0.0:
    raise ValueError(f'dt must be positive, got {dt}')
  return dt


@dataclass(frozen=True)
class PhysicalScales:
  """Reference scales for quoting growth rates physically, all SI.

  Lengths use the current-sheet half-width `lam_b`, density the sheet peak
  `n0`, and field the asymptotic `B0 = psi0` of the Harris profile.
  """

  deck: str
  unit_convention: str
  reference_length_m: float
  reference_density_m3: float
  ion_mass_kg: float
  field_T: float
  magnetic_diffusivity_m2_s: float
  kinematic_viscosity_m2_s: float
  alfven_speed_m_s: float
  alfven_time_s: float
  lundquist_number: float
  magnetic_prandtl_number: float

  def describe(self) -> dict[str, Any]:
    """A JSON-ready dictionary."""
    return asdict(self)


#: How each deck generation's driver reads `eta` and `nu`.
UNIT_CONVENTIONS = {
  'harris_sheet_options': 'eta is magnetic diffusivity [m^2/s]; nu is kinematic viscosity [m^2/s]',
  'xmhd_options': 'eta is resistivity [Ohm m]; nu is dynamic viscosity [Pa s], taken at n0',
}


def read_physical_scales(deck_path: str) -> PhysicalScales:
  """Alfven and dissipation scales of a Harris-sheet deck.

  The two drivers read `eta` and `nu` in different units, so the deck's
  group decides the conversion: the older linear driver divided its
  resistivity by mu0 and its viscosity by the mass density in the equations;
  upstream OFT takes both as diffusivities.
  """
  with open(deck_path, encoding='utf-8', errors='replace') as f:
    text = f.read()
  group = simulation_namelist(text)

  def value(key: str) -> float:
    number = float(read_namelist_value(text, group, key).lower().replace('d', 'e'))
    if number <= 0.0:
      raise ValueError(f'{key} must be positive to define physical scales, got {number}')
    return number

  field, length = value('psi0'), value('lam_b')
  density, eta, nu = value('n0'), value('eta'), value('nu')
  mass_density = PROTON_MASS * density
  if group == 'xmhd_options':
    diffusivity, viscosity = eta / MU0, nu / mass_density
  else:
    diffusivity, viscosity = eta, nu

  alfven_speed = field / math.sqrt(MU0 * mass_density)
  return PhysicalScales(
    deck=os.path.abspath(deck_path),
    unit_convention=UNIT_CONVENTIONS[group],
    reference_length_m=length,
    reference_density_m3=density,
    ion_mass_kg=PROTON_MASS,
    field_T=field,
    magnetic_diffusivity_m2_s=diffusivity,
    kinematic_viscosity_m2_s=viscosity,
    alfven_speed_m_s=alfven_speed,
    alfven_time_s=length / alfven_speed,
    lundquist_number=length * alfven_speed / diffusivity,
    magnetic_prandtl_number=viscosity / diffusivity,
  )
