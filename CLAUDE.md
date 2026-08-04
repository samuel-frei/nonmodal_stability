# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`nonmodal` computes **nonmodal (pseudospectral) stability diagnostics** for reduced
linear MHD operators. It loads exported Jacobian and mass matrices, reduces the
global system to a subset of physical fields, builds the Schur form of the
time-advance operator, and samples `sigma_min(zI - T)` across the complex plane.

It is a **library that also ships a console command** — the notebook imports it, and
the SLURM decks run it.

## Commands

```bash
uv sync                                          # create .venv (Python 3.14+)
uv run ruff check . && uv run mypy && uv run pytest   # the full local gate
uv run pytest tests/test_pseudospectrum.py -k closed_form   # a single test
uv run nonmodal --help                           # console script
uv run python -m nonmodal --help                 # equivalent
```

CI (`.github/workflows/ci.yml`) runs ruff, mypy and pytest on 3.14 against `origin`
(the private `nonmodal_stability` repo).

## Provenance — important

This repo **is** `git@github.com:samuel-frei/nonmodal_stability`, restructured into a
package. It was grafted onto that repo's 18-commit history rather than started fresh,
so `git log --follow` traces the original script through the moves.

Two stale copies exist outside this repo and are **not** authoritative:

- `research/recon_data/scripts/lin_ops_full_reduction.py` — equals upstream commit
  `0d0ae3a`, three commits behind, and still contains adaptive-grid code that upstream
  deliberately deleted in `304ff7f` ("remove naive mesh refinement").
- `research/recon_data/psc_data/scripts/lin_ops_full_reduction.py` — was *ahead* of
  upstream; its external-grid work has since been merged here (commit `ef52242`).

This repo is ground truth. Do not treat those copies as a compatibility constraint.

## Architecture

Pipeline stages, in `run_pipeline` ([pipeline.py](src/nonmodal/pipeline.py)):

1. **Reduce** — `build_reduction_mapping` builds the keep-mask;
   `load_or_compute_jacobian` assembles global matrices and forms the effective
   operator `real_jac = (spsolve(reduced_jac, reduced_mmat) - I) / DEFAULT_TIMESTEP`.
   This turns the implicit time-advance into a linear operator worth analysing.
2. **Factorise** — full eigenvalues, 40 eigenvectors (written out as `.rst` restart
   files), and the complex Schur factor `T` used for all sampling.
3. **Sample** — for each grid point `z`, inverse-power iteration via `eigsh` on a
   `LinearOperator` applying `(zI - T)^-* (zI - T)^-1` through LAPACK `trtrs`
   triangular solves. `zI - T` is never materialised densely.
4. **Emit** — `.npy` arrays, Plotly HTML, `run_metadata.json`.

### Module map

| Module | Responsibility |
|---|---|
| `matrices.py` | `HDF5Matrix`, `assemble_global` (numba) |
| `fields.py` | 7-field block layout, reduction mask, restart output |
| `operator.py` | reduced operator, eigenvalues, Schur factor, caching |
| `grid.py` | grid geometry, flat-grid loading |
| `pseudospectrum.py` | sigma_min sampling, fork-pool parallelism, contour levels |
| `plotting.py` | Plotly heatmap and contour views |
| `io.py` | run metadata, array output |
| `pipeline.py` | `run_pipeline`, input validation |
| `cli.py` | argparse surface, `main` |

### Naming rule

**Do not use `oft` / `OFT` in our identifiers.** Those names are reserved for genuine
OpenFUSIONToolkit interop (a future `adapters.py`), so that a local module never sits
confusingly beside the real package. This is why the HDF5 reader is `HDF5Matrix` and
the block layout lives in `fields.py` rather than a vendor-named module.

## Landmines

- **Thread pinning must stay at the top of `__init__.py`**, above every import.
  `os.environ.setdefault` for `OMP/MKL/OPENBLAS_NUM_THREADS` only takes effect before
  NumPy is first imported. Moving or reordering it silently degrades many-core runs
  instead of erroring. It uses `setdefault` so a caller who sets threads *before*
  importing (the notebook does, wanting 16) keeps their choice.
- **Fork worker globals.** `_worker_T` / `_worker_trtrs` in `pseudospectrum.py` are set
  in the parent and inherited via `fork`. They must stay in the same module as
  `_compute_sig_point`, and be set on that module object, not rebound locally.
- **Caches are keyed by filename only.** `real_jacobian.npy`, `full_reduced_*.npy` under
  `--cache-dir` are reused blindly; they are *not* invalidated when inputs,
  `DEFAULT_TIMESTEP` or `KEPT_BLOCK_IDS` change. Delete them by hand. Cache hits log
  path and mtime so stale reuse is visible in the log.
- **Half-plane mirroring is conditional.** `compute_pseudospectrum` evaluates only the
  upper half-plane when the imaginary axis is symmetric about zero, then mirrors. Valid
  only for conjugate-symmetric spectra (true for the real reduced operator). An
  arbitrary complex operator needs asymmetric imaginary bounds to force a full grid.
- **`assemble_global` uses `inmat.shape[0]` for both loop bounds** — square input only.
  It now raises rather than silently mis-assembling.
- **Plots use `include_plotlyjs='cdn'`** and render blank offline.

## Testing approach

The load-bearing test is exact, not golden-file: for a diagonal (already Schur-form)
operator, `sigma_min(zI - T) = min_i |z - T_ii|` in closed form, so the LAPACK/eigsh
path is verified analytically on both the mirrored and full-grid branches.

`tests/conftest.py` forces the Agg backend — `operator.py` imports pyplot at module
scope.
