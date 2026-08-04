# nonmodal

Nonmodal (pseudospectral) stability analysis of reduced linear MHD operators.

Given exported Jacobian and mass matrices, `nonmodal` reduces the global system
to a subset of physical fields, builds the Schur form of the resulting
time-advance operator, and samples the resolvent norm

$$\sigma_{\min}(zI - T)$$

over a region of the complex plane. The $\varepsilon$-pseudospectrum contours
that result show how far the operator's response can be amplified by
perturbations that a purely modal (eigenvalue) analysis would call stable.

It is both a library and a command-line tool, and is built for HPC batch
execution: sampling is parallelised across a process pool, and every stage
caches its intermediate result.

## Install

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Inputs

A run needs two HDF5 files produced by the simulation:

| File | Dataset group | Contents |
|---|---|---|
| `lin_ops.h5` | `/jacobian` | linearised Jacobian |
| `mass_mat.h5` | `/massmat` | mass matrix (one field block) |

Index arrays inside them are one-based and converted on load.

### Field-block layout

The global state vector is seven equally sized blocks:

```
U_n  U_velx  U_vely  U_velz  U_T  U_psi  U_by
 0     1       2       3      4     5      6
```

Reduction keeps blocks `(0, 1, 3, 4, 5)` — `n, velx, velz, T, psi` — dropping
`vely` and `by`, and additionally drops boundary-condition rows. See
`KEPT_BLOCK_IDS` in `src/nonmodal/fields.py`.

## Command-line use

Run from a directory containing the input files:

```bash
uv run nonmodal --grid-points 3200 --nprocs 32 --real-min -9e6 --real-max 5e5 --imag-min -3e6 --imag-max 3e6
```

`python -m nonmodal` works identically. See `cases/harris_linear_20x6z/ps1.sbatch`
for a portable SLURM template.

Instead of rectangular bounds you can supply a precomputed flat complex grid
with `--grid-npy grid.npy`; add `--grid-shape ROWS COLS` if you also want plots.

### Outputs

Written to `--output-dir` (default `pseudospectrum/`):

- `pseudo_R.npy`, `pseudo_C.npy`, `pseudo_sigmin.npy` — sampled grid and values
  (or `pseudo_z.npy` / `pseudo_sigmin_flat.npy` for an unstructured grid)
- `*_heatmap.html`, `*_contours.html` — interactive Plotly views
- `run_metadata.json` — bounds, grid, worker counts, resolved input paths
- `eigvecs_plot/xmhd2d_*.rst` — leading eigenvectors as restart files

> **Note:** plots are written with `include_plotlyjs='cdn'`, so they render
> blank on a machine without internet access. Open them somewhere with network
> access, or change the call in `src/nonmodal/plotting.py` to
> `include_plotlyjs=True` for self-contained files.

## Caching — read this before trusting a result

Four expensive intermediates are cached under `--cache-dir` (default `.`):

```
real_jacobian.npy  full_reduced_eigvals.npy
full_reduced_schur.npy  full_reduced_eigvecs.npy
```

**Caches are keyed by filename alone.** They are *not* invalidated when the
input matrices, `DEFAULT_TIMESTEP`, or `KEPT_BLOCK_IDS` change. Changing any of
those while stale caches are present will silently produce results for the
*previous* problem.

Delete them by hand after any such change:

```bash
rm -f real_jacobian.npy full_reduced_*.npy
```

Every cache hit logs its absolute path and mtime so you can spot a stale reuse
in the run log.

## Library use

```python
import numpy as np
from nonmodal import HDF5Matrix, build_reduction_mapping, compute_pseudospectrum

nr_local, keep_global = build_reduction_mapping('lin_ops.h5')
jac = HDF5Matrix('lin_ops.h5', '/jacobian')

R, C, sigmin = compute_pseudospectrum(
    schur_t, grid_points=1024, nprocs=16,
    real_min=-9e6, real_max=5e5, imag_min=-3e6, imag_max=3e6)
```

The package ships `py.typed`.

Two behaviours worth knowing:

- **Half-plane mirroring.** When the imaginary axis is symmetric about zero,
  only the upper half is evaluated and mirrored. That is valid only for
  operators whose spectrum is closed under conjugation (true for the real
  reduced operator). For an arbitrary complex operator, pass asymmetric
  imaginary bounds to force full-grid evaluation.
- **BLAS threads.** Importing `nonmodal` pins `OMP/MKL/OPENBLAS_NUM_THREADS` to
  `1`, because the sampling pool would otherwise oversubscribe every core. It
  uses `setdefault`, so if you want more threads for your own dense linear
  algebra, set them *before* importing.

## Development

```bash
uv run ruff check . && uv run mypy && uv run pytest
```

## Repository layout

```
src/nonmodal/   the library
cases/          OFT input decks and submission scripts
notebooks/      exploratory analysis
tests/
```

`cases/` holds the simulation inputs that generate the `.h5` files this package
consumes; `run_script.sh` in each case invokes the solver itself, not this code.
