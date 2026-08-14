"""Rendering scattered samples: filename helpers, interpolation, HTML output."""

import re

import numpy as np
import pytest

from nonmodal.plotting import (
  _normalize_html_name,
  _safe_log10,
  _split_plot_output_names,
  interpolate_to_mesh,
  pseudo_contours,
  pseudo_heatmap,
)


def test_normalize_html_name_adds_suffix() -> None:
  assert _normalize_html_name('plot', 'plot-name') == 'plot.html'
  assert _normalize_html_name('plot.html', 'plot-name') == 'plot.html'
  assert _normalize_html_name('  plot.HTML  ', 'plot-name') == 'plot.HTML'


def test_normalize_html_name_rejects_empty() -> None:
  with pytest.raises(ValueError, match='must not be empty'):
    _normalize_html_name('   ', 'plot-name')


def test_split_plot_output_names() -> None:
  assert _split_plot_output_names('p.html') == ('p_heatmap.html', 'p_contours.html')
  assert _split_plot_output_names('p') == ('p_heatmap.html', 'p_contours.html')


def test_safe_log10_masks_nonpositive() -> None:
  out = _safe_log10(np.array([1.0, 0.0, -1.0, np.nan, 100.0]))
  assert out[0] == pytest.approx(0.0)
  assert out[4] == pytest.approx(2.0)
  assert np.isnan(out[1:4]).all()


def _log_linear(z: np.ndarray) -> np.ndarray:
  """A field whose log10 is exactly linear in (Re z, Im z).

  Linear interpolation is then exact, to machine precision.
  """
  return np.asarray(10.0 ** (0.5 * z.real + 0.25 * z.imag))


@pytest.fixture
def samples():
  """A scattered sample set covering the unit box."""
  rng = np.random.default_rng(0)
  z = (rng.uniform(-1, 1, 400) + 1j * rng.uniform(-1, 1, 400)).astype(np.complex128)
  # Include the corners so the convex hull covers the whole box.
  z = np.concatenate([z, np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j])])
  eigvals = np.array([0.1 + 0.2j, -0.3 - 0.4j], dtype=np.complex128)
  return z, _log_linear(z), eigvals


def test_interpolate_to_mesh_is_exact_for_a_log_linear_field(samples) -> None:
  z, sigmin, _ = samples
  x, y, log_sig = interpolate_to_mesh(z, sigmin, mesh=40)

  assert x.shape == (40,) and y.shape == (40,)
  assert log_sig.shape == (40, 40)

  X, Y = np.meshgrid(x, y)
  expected = 0.5 * X + 0.25 * Y
  covered = np.isfinite(log_sig)
  assert covered.sum() > 1000
  np.testing.assert_allclose(log_sig[covered], expected[covered], atol=1e-12)


def test_interpolate_rejects_too_few_samples() -> None:
  z = np.array([0 + 0j, 1 + 0j], dtype=np.complex128)
  with pytest.raises(ValueError, match='at least 3'):
    interpolate_to_mesh(z, np.array([1.0, 2.0]), mesh=8)


def test_heatmap_writes_html(tmp_path, samples) -> None:
  z, sigmin, eigvals = samples
  out = pseudo_heatmap(str(tmp_path), 'h.html', z, sigmin, eigvals, mesh=32)
  assert out.endswith('h.html')
  assert (tmp_path / 'h.html').stat().st_size > 0


def test_contours_written_from_samples(tmp_path, samples) -> None:
  z, sigmin, eigvals = samples
  # All four levels sit inside the field's range, so all four get drawn.
  levels = np.geomspace(0.3, 3.0, 4)
  out = pseudo_contours(str(tmp_path), 'c.html', z, sigmin, eigvals, levels)
  assert out.endswith('c.html')

  html = (tmp_path / 'c.html').read_text()
  names = set(re.findall(r'"name":"([^"]+)"', html))
  # Each level is labelled in epsilon units rather than log10.
  for level in levels:
    assert f'{level:.2e}' in names, f'no trace labelled for level {level:.2e}'


def test_contour_labels_survive_levels_matplotlib_drops(tmp_path, samples) -> None:
  """Levels outside the data range are dropped, and the rest stay correctly labelled."""
  z, sigmin, eigvals = samples
  in_range = 1.0
  levels = np.array([1e-6, in_range, 1e6])  # first and last lie outside

  pseudo_contours(str(tmp_path), 'c.html', z, sigmin, eigvals, levels)
  names = set(re.findall(r'"name":"([^"]+)"', (tmp_path / 'c.html').read_text()))
  assert f'{in_range:.2e}' in names
  assert '1.00e-06' not in names and '1.00e+06' not in names


def test_inline_js_embeds_plotly(tmp_path, samples) -> None:
  z, sigmin, eigvals = samples
  cdn = pseudo_heatmap(str(tmp_path), 'cdn.html', z, sigmin, eigvals, mesh=16)
  inline = pseudo_heatmap(
    str(tmp_path), 'inline.html', z, sigmin, eigvals, mesh=16, inline_js=True)

  # The inline page carries the library itself, so it is far larger.
  assert len(open(inline).read()) > 10 * len(open(cdn).read())


def test_contours_reject_bad_levels(tmp_path, samples) -> None:
  z, sigmin, eigvals = samples
  with pytest.raises(ValueError, match='at least one contour value'):
    pseudo_contours(str(tmp_path), 'c.html', z, sigmin, eigvals, np.array([]))
  with pytest.raises(ValueError, match='strictly positive'):
    pseudo_contours(str(tmp_path), 'c.html', z, sigmin, eigvals, np.array([0.0, 1.0]))


def test_contours_need_three_points(tmp_path) -> None:
  z = np.array([0 + 0j, 1 + 0j], dtype=np.complex128)
  with pytest.raises(ValueError, match='at least 3 sample points'):
    pseudo_contours(
      str(tmp_path), 'c.html', z, np.array([1.0, 2.0]),
      np.zeros(0, dtype=np.complex128), np.array([1.0]))
