"""Plot filename helpers and HTML output smoke tests."""

import numpy as np
import pytest

from nonmodal.plotting import (
  _normalize_html_name,
  _split_plot_output_names,
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


@pytest.fixture
def sample_field():
  r = np.linspace(-1.0, 1.0, 8)
  c = np.linspace(-1.0, 1.0, 8)
  R, C = np.meshgrid(r, c)
  sigmin = np.abs(R + 1j * C) + 1e-3
  eigvals = np.array([0.1 + 0.2j, -0.3 - 0.4j], dtype=np.complex128)
  return R, C, sigmin, eigvals


def test_heatmap_writes_html(tmp_path, sample_field) -> None:
  R, C, sigmin, eigvals = sample_field
  out = pseudo_heatmap(str(tmp_path), 'h.html', R, C, sigmin, eigvals)
  assert out.endswith('h.html')
  assert (tmp_path / 'h.html').stat().st_size > 0


def test_contours_writes_html(tmp_path, sample_field) -> None:
  R, C, sigmin, eigvals = sample_field
  levels = np.geomspace(1e-2, 1.0, 4)
  out = pseudo_contours(str(tmp_path), 'c.html', R, C, sigmin, eigvals, levels)
  assert out.endswith('c.html')
  assert (tmp_path / 'c.html').stat().st_size > 0


def test_contours_reject_bad_levels(tmp_path, sample_field) -> None:
  R, C, sigmin, eigvals = sample_field
  with pytest.raises(ValueError, match='at least one contour value'):
    pseudo_contours(str(tmp_path), 'c.html', R, C, sigmin, eigvals, np.array([]))
  with pytest.raises(ValueError, match='strictly positive'):
    pseudo_contours(str(tmp_path), 'c.html', R, C, sigmin, eigvals, np.array([0.0, 1.0]))
