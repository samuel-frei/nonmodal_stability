"""Interactive Plotly views of a sampled pseudospectrum."""

import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

TITLE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 14
COLORBAR_TITLE_FONT_SIZE = 18
COLORBAR_TICK_FONT_SIZE = 14
CONTOUR_LABEL_FONT_SIZE = 12


def _normalize_html_name(name: str, label: str) -> str:
  """Normalise an output filename to carry an .html suffix."""
  val = str(name).strip()
  if not val:
    raise ValueError(f'{label} must not be empty')
  if not val.lower().endswith('.html'):
    fixed = f'{val}.html'
    print(f'adjusted {label} to HTML output: {fixed}', flush=True)
    return fixed
  return val


def _split_plot_output_names(plot_name: str) -> tuple[str, str]:
  """Derive separate output names for the heatmap and contour views."""
  stem, ext = os.path.splitext(str(plot_name))
  ext = ext if ext else '.html'
  return f'{stem}_heatmap{ext}', f'{stem}_contours{ext}'


def _add_eigenvalue_overlay(fig: Any, eigvals: NDArray[np.complexfloating]) -> None:
  """Overlay eigenvalue markers on an existing Plotly figure."""
  eigvals_arr = np.asarray(eigvals).ravel()
  if eigvals_arr.size > 0:
    mask = np.isfinite(eigvals_arr.real) & np.isfinite(eigvals_arr.imag)
    eigvals_arr = eigvals_arr[mask]
  if eigvals_arr.size > 0:
    fig.add_trace(go.Scatter(
      x=eigvals_arr.real,
      y=eigvals_arr.imag,
      mode='markers',
      name='eigenvalues',
      marker={'size': 2.5, 'color': 'black', 'opacity': 0.65},
      hovertemplate='Re[lambda]=%{x:.6g}<br>Im[lambda]=%{y:.6g}<extra></extra>'))


def _apply_common_layout(
  fig: Any,
  title: str,
  x_min: float,
  x_max: float,
  y_min: float,
  y_max: float,
) -> None:
  """Apply the shared axis, title and margin styling to a figure."""
  fig.update_layout(
    title={
      'text': title,
      'x': 0.5,
      'xanchor': 'center',
      'font': {'size': TITLE_FONT_SIZE},
    },
    xaxis={
      'title': {'text': 'Re[z]', 'font': {'size': AXIS_TITLE_FONT_SIZE}},
      'tickfont': {'size': AXIS_TICK_FONT_SIZE},
      'tickformat': '.3g',
      'constrain': 'domain',
      'autorange': False,
      'range': [x_min, x_max],
    },
    yaxis={
      'title': {'text': 'Im[z]', 'font': {'size': AXIS_TITLE_FONT_SIZE}},
      'tickfont': {'size': AXIS_TICK_FONT_SIZE},
      'tickformat': '.3g',
      'autorange': False,
      'range': [y_min, y_max],
    },
    margin={'l': 60, 'r': 200, 'b': 55, 't': 50},
    dragmode='zoom')


def _axis_extents(
  R: NDArray[np.float64],
  C: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float, float]:
  """Extract 1-D axes and their extents from meshgrid coordinate arrays."""
  x_axis = np.asarray(R, dtype=float)[0, :]
  y_axis = np.asarray(C, dtype=float)[:, 0]
  return (
    x_axis,
    y_axis,
    float(np.min(x_axis)),
    float(np.max(x_axis)),
    float(np.min(y_axis)),
    float(np.max(y_axis)))


def pseudo_heatmap(
  output_dir: str,
  plot_name: str,
  R: NDArray[np.float64],
  C: NDArray[np.float64],
  sigmin: NDArray[np.float64],
  eigvals: NDArray[np.complexfloating],
) -> str:
  """Render an interactive heatmap view of pseudospectrum values."""
  x_axis, y_axis, x_min, x_max, y_min, y_max = _axis_extents(R, C)

  sig_values = np.asarray(sigmin, dtype=float)
  positive_finite = np.isfinite(sig_values) & (sig_values > 0.0)
  log_sig = np.full(sig_values.shape, np.nan, dtype=float)
  if np.any(positive_finite):
    log_sig[positive_finite] = np.log10(sig_values[positive_finite])

  finite_log = np.isfinite(log_sig)
  if not np.any(finite_log):
    tickvals = np.array([0.0], dtype=float)
    ticktext = ['nan']
    zmin = 0.0
    zmax = 0.0
  else:
    zmin = float(np.nanmin(log_sig[finite_log]))
    zmax = float(np.nanmax(log_sig[finite_log]))
    if zmax <= zmin:
      zmax = np.nextafter(zmin, np.inf)
    tickvals = np.linspace(zmin, zmax, 6)
    ticktext = [f'{10.0 ** val:.3e}' for val in tickvals]

  fig = go.Figure(go.Heatmap(
    x=x_axis,
    y=y_axis,
    z=log_sig,
    zmin=zmin,
    zmax=zmax,
    zsmooth='best',
    colorscale='Viridis',
    colorbar={
      'title': {
        'text': 'ε',
        'font': {'size': COLORBAR_TITLE_FONT_SIZE},
      },
      'tickmode': 'array',
      'tickvals': tickvals,
      'ticktext': ticktext,
      'tickfont': {'size': COLORBAR_TICK_FONT_SIZE},
      'ticks': '',
      'ticklen': 0,
    },
    hovertemplate='Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>log10(epsilon)=%{z:.4f}<extra></extra>'))

  _add_eigenvalue_overlay(fig, eigvals)
  _apply_common_layout(
    fig, 'Pseudospectra Heatmap of Resistive MHD Operator', x_min, x_max, y_min, y_max)

  out_path = os.path.join(output_dir, plot_name)
  fig.write_html(out_path, include_plotlyjs='cdn')
  print(f'wrote interactive heatmap: {out_path}', flush=True)
  return out_path


def pseudo_contours(
  output_dir: str,
  plot_name: str,
  R: NDArray[np.float64],
  C: NDArray[np.float64],
  sigmin: NDArray[np.float64],
  eigvals: NDArray[np.complexfloating],
  levels: NDArray[np.float64],
) -> str:
  """Render an interactive, colour-coded contour view of pseudospectrum values."""
  levels = np.asarray(levels, dtype=float).ravel()
  if levels.size == 0:
    raise ValueError('levels must contain at least one contour value')
  levels = np.unique(np.sort(levels))
  if levels[0] <= 0.0:
    raise ValueError('levels must be strictly positive')

  # Keep log-space color progression for wide dynamic range while directly
  # labeling contour lines in epsilon units.
  log_levels = np.log10(levels)
  contour_start = float(log_levels[0])
  contour_end = float(log_levels[-1])

  x_axis, y_axis, x_min, x_max, y_min, y_max = _axis_extents(R, C)

  sig_values = np.asarray(sigmin, dtype=float)
  finite_positive = np.isfinite(sig_values) & (sig_values > 0.0)
  safe_sig = np.array(sig_values, copy=True)
  safe_sig[~finite_positive] = np.nan
  safe_log_sig = np.array(sig_values, copy=True)
  safe_log_sig[finite_positive] = np.log10(sig_values[finite_positive])
  safe_log_sig[~finite_positive] = np.nan

  level_tick_vals = np.unique(np.concatenate(([contour_start, contour_end], log_levels)))
  level_tick_text = [f'{10.0 ** val:.1e}' for val in level_tick_vals]

  if levels.size == 1:
    contour_end = float(np.nextafter(contour_start, np.inf))
    contour_size = float(contour_end - contour_start)
  else:
    contour_size = float((contour_end - contour_start) / (levels.size - 1))
  if contour_size <= 0.0:
    contour_size = float(np.nextafter(0.0, 1.0))

  fig = go.Figure()
  fig.add_trace(go.Contour(
    x=x_axis,
    y=y_axis,
    z=safe_log_sig,
    customdata=safe_sig,
    zmin=contour_start,
    zmax=contour_end,
    autocontour=False,
    colorscale='Turbo',
    contours={
      'start': contour_start,
      'end': contour_end,
      'size': contour_size,
      'showlabels': False,
      'coloring': 'lines',
    },
    line={'width': 2.0},
    colorbar={
      'title': {
        'text': 'ε',
        'font': {'size': COLORBAR_TITLE_FONT_SIZE},
      },
      'tickmode': 'array',
      'tickvals': level_tick_vals,
      'ticktext': level_tick_text,
      'tickfont': {'size': COLORBAR_TICK_FONT_SIZE},
      'ticks': '',
      'ticklen': 0,
    },
    hovertemplate=(
      'Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>'
      'epsilon=%{customdata:.4e}<br>log10(epsilon)=%{z:.4f}<extra></extra>')))

  # Overlay one contour trace per level to label lines directly in epsilon units.
  for level_value in levels:
    level = float(level_value)
    label_end = float(np.nextafter(level, np.inf))
    label_size = float(max(label_end - level, np.finfo(float).eps))
    fig.add_trace(go.Contour(
      x=x_axis,
      y=y_axis,
      z=safe_sig,
      autocontour=False,
      showscale=False,
      hoverinfo='skip',
      contours={
        'start': level,
        'end': label_end,
        'size': label_size,
        'showlabels': True,
        'labelformat': '.2e',
        'labelfont': {'size': CONTOUR_LABEL_FONT_SIZE, 'color': 'black'},
        'coloring': 'none',
      },
      line={'width': 0.0, 'color': 'rgba(0,0,0,0)'}))

  _add_eigenvalue_overlay(fig, eigvals)
  _apply_common_layout(
    fig, 'Pseudospectra Contours of Resistive MHD Operator', x_min, x_max, y_min, y_max)

  out_path = os.path.join(output_dir, plot_name)
  fig.write_html(out_path, include_plotlyjs='cdn')
  print(f'wrote interactive contours: {out_path}', flush=True)
  return out_path
