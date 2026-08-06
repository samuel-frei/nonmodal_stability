"""Interactive Plotly views built from scattered pseudospectrum samples.

Samples arrive as a flat point set, so structure is reintroduced here:

* contours come from `matplotlib.tri.tricontour` on a Delaunay triangulation of
  the actual samples, so no interpolation error is introduced where it matters;
* the heatmap interpolates onto a regular mesh, because a raster needs one.

Both work in log10(sigma_min). Interpolating the raw value would misplace
exactly the contours of interest: sigma_min spans orders of magnitude, dropping
to ~3e-4 of the eigenvalue distance on a strongly non-normal operator.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator

TITLE_FONT_SIZE = 20
AXIS_TITLE_FONT_SIZE = 18
AXIS_TICK_FONT_SIZE = 14
COLORBAR_TITLE_FONT_SIZE = 18
COLORBAR_TICK_FONT_SIZE = 14


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


def _safe_log10(sigmin: NDArray[np.float64]) -> NDArray[np.float64]:
  """log10 of positive finite values, NaN elsewhere."""
  values = np.asarray(sigmin, dtype=float)
  out = np.full(values.shape, np.nan, dtype=float)
  usable = np.isfinite(values) & (values > 0.0)
  out[usable] = np.log10(values[usable])
  return out


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
  fig: Any, title: str, z: NDArray[np.complex128]
) -> None:
  """Apply shared axis, title and margin styling, framed on the sample set."""
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
      'range': [float(z.real.min()), float(z.real.max())],
    },
    yaxis={
      'title': {'text': 'Im[z]', 'font': {'size': AXIS_TITLE_FONT_SIZE}},
      'tickfont': {'size': AXIS_TICK_FONT_SIZE},
      'tickformat': '.3g',
      'autorange': False,
      'range': [float(z.imag.min()), float(z.imag.max())],
    },
    margin={'l': 60, 'r': 200, 'b': 55, 't': 50},
    dragmode='zoom')


def _write(fig: Any, output_dir: str, plot_name: str, inline_js: bool, what: str) -> str:
  os.makedirs(output_dir, exist_ok=True)
  out_path = os.path.join(output_dir, plot_name)
  fig.write_html(out_path, include_plotlyjs=True if inline_js else 'cdn')
  print(f'wrote interactive {what}: {out_path}', flush=True)
  return out_path


def _triangulation(z: NDArray[np.complex128]) -> mtri.Triangulation:
  if z.size < 3:
    raise ValueError('at least 3 sample points are needed to triangulate')
  return mtri.Triangulation(z.real, z.imag)


def pseudo_contours(
  output_dir: str,
  plot_name: str,
  z: NDArray[np.complex128],
  sigmin: NDArray[np.float64],
  eigvals: NDArray[np.complexfloating],
  levels: NDArray[np.float64],
  inline_js: bool = False,
) -> str:
  """Render contours drawn directly from the sampled points."""
  levels = np.unique(np.sort(np.asarray(levels, dtype=float).ravel()))
  if levels.size == 0:
    raise ValueError('levels must contain at least one contour value')
  if levels[0] <= 0.0:
    raise ValueError('levels must be strictly positive')

  tri = _triangulation(z)
  values = _safe_log10(sigmin)
  # tricontour cannot handle NaN vertices; mask any triangle touching one.
  bad = ~np.isfinite(values)
  if bad.any():
    tri.set_mask(bad[tri.triangles].any(axis=1))
    values = np.nan_to_num(values, nan=float(np.nanmin(values)))

  fig = go.Figure()

  # tricontour needs an Axes to draw into; the figure is never rendered, only
  # mined for its contour vertices.
  figure = plt.figure()
  try:
    ax = figure.add_subplot(111)
    cs = ax.tricontour(tri, values, levels=np.log10(levels))
    # Pair against cs.levels rather than the requested levels: matplotlib may
    # drop levels that fall outside the data range, and zipping the requested
    # list against allsegs would then label contours with the wrong epsilon.
    drawn = np.power(10.0, np.asarray(cs.levels, dtype=float))
    palette = _level_colours(len(drawn))
    for level_value, colour, segs in zip(
      drawn, palette, cs.allsegs, strict=True
    ):
      first = True
      for seg in segs:
        if seg.shape[0] < 2:
          continue
        fig.add_trace(go.Scatter(
          x=seg[:, 0],
          y=seg[:, 1],
          mode='lines',
          line={'width': 2.0, 'color': colour},
          name=f'{level_value:.2e}',
          legendgroup=f'{level_value:.2e}',
          showlegend=first,
          hovertemplate=(
            f'epsilon={level_value:.4e}<br>'
            'Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<extra></extra>')))
        first = False
  finally:
    plt.close(figure)

  _add_eigenvalue_overlay(fig, eigvals)
  _apply_common_layout(fig, 'Pseudospectra Contours of Resistive MHD Operator', z)
  fig.update_layout(legend={'title': {'text': 'ε'}})
  return _write(fig, output_dir, plot_name, inline_js, 'contours')


def _level_colours(n: int) -> list[str]:
  """Sample the Turbo colourscale at n points."""
  if n == 1:
    return [pc.sample_colorscale('Turbo', [0.5])[0]]
  return pc.sample_colorscale('Turbo', list(np.linspace(0.0, 1.0, n)))


def interpolate_to_mesh(
  z: NDArray[np.complex128],
  sigmin: NDArray[np.float64],
  mesh: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
  """Interpolate log10(sigma_min) from scattered samples onto a regular mesh."""
  values = _safe_log10(sigmin)
  usable = np.isfinite(values)
  if usable.sum() < 3:
    raise ValueError('need at least 3 positive finite samples to interpolate')

  x = np.linspace(float(z.real.min()), float(z.real.max()), mesh)
  y = np.linspace(float(z.imag.min()), float(z.imag.max()), mesh)
  X, Y = np.meshgrid(x, y)
  interp = LinearNDInterpolator(
    np.column_stack([z.real[usable], z.imag[usable]]), values[usable])
  return x, y, np.asarray(interp(X, Y), dtype=float)


def pseudo_heatmap(
  output_dir: str,
  plot_name: str,
  z: NDArray[np.complex128],
  sigmin: NDArray[np.float64],
  eigvals: NDArray[np.complexfloating],
  mesh: int = 400,
  inline_js: bool = False,
) -> str:
  """Render a heatmap by interpolating the samples onto a regular mesh."""
  x_axis, y_axis, log_sig = interpolate_to_mesh(z, sigmin, mesh)

  finite = np.isfinite(log_sig)
  if not finite.any():
    zmin, zmax = 0.0, 1.0
    tickvals = np.array([0.0])
    ticktext = ['nan']
  else:
    zmin = float(np.nanmin(log_sig[finite]))
    zmax = float(np.nanmax(log_sig[finite]))
    if zmax <= zmin:
      zmax = float(np.nextafter(zmin, np.inf))
    tickvals = np.linspace(zmin, zmax, 6)
    ticktext = [f'{10.0 ** val:.3e}' for val in tickvals]

  fig = go.Figure(go.Heatmap(
    x=x_axis,
    y=y_axis,
    z=log_sig,
    zmin=zmin,
    zmax=zmax,
    colorscale='Viridis',
    colorbar={
      'title': {'text': 'ε', 'font': {'size': COLORBAR_TITLE_FONT_SIZE}},
      'tickmode': 'array',
      'tickvals': tickvals,
      'ticktext': ticktext,
      'tickfont': {'size': COLORBAR_TICK_FONT_SIZE},
      'ticks': '',
      'ticklen': 0,
    },
    hovertemplate='Re[z]=%{x:.6g}<br>Im[z]=%{y:.6g}<br>log10(epsilon)=%{z:.4f}<extra></extra>'))

  _add_eigenvalue_overlay(fig, eigvals)
  _apply_common_layout(fig, 'Pseudospectra Heatmap of Resistive MHD Operator', z)
  return _write(fig, output_dir, plot_name, inline_js, 'heatmap')
