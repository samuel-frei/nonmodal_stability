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
uv run nonmodal run --help                       # sample a pseudospectrum
uv run nonmodal plot --help                      # render a finished run
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

## Where the real data lives

Nothing in this repo; the exports sit under `research/recon_data/psc_data/`.

- `harris_linear_20x6z/{pseudo_marginal,pseudo_hires,pseudo_hhres,pseudo_hhhres,
  margin_finding}/` — the production runs, Apr 2026. Reduced operator is
  **10500x10500** (`real_jacobian.npy` 882 MB, `full_reduced_schur.npy` 1.76 GB).
  `pseudo_hhhres` is newest and has plots.
- Those runs saved `pseudo_R/C/sigmin.npy` — the **old structured mesh**.
  `nonmodal plot` expects flat `pseudo_z.npy` + `pseudo_sigmin.npy`, so it cannot
  read them without converting: `z = (R + 1j*C).ravel()`.
- `harris_linear/lin_ops.h5` is **not loadable** by `HDF5Matrix`: it lacks
  `nrg`, `ncg` and `lg`, and its dimensions (51247 = 7x7321) say it is a
  pre-assembled *global* matrix rather than the local-plus-scatter layout the
  reader assumes. The `20x6z` exports are the ones that match.

## Architecture

**Samples are flat everywhere.** A sample set is a 1-D complex array of points plus a
1-D array of values — never a mesh. Structure is reintroduced only at plot time, by
triangulating or interpolating. That is what keeps the pipeline free of the
structured/unstructured branching it used to carry.

`nonmodal run` ([pipeline.py](src/nonmodal/pipeline.py)):

1. **Reduce** — `build_reduction_mapping` builds the keep-mask;
   `load_or_compute_jacobian` assembles global matrices and forms the effective
   operator `real_jac = (spsolve(reduced_jac, reduced_mmat) - I) / DEFAULT_TIMESTEP`.
   This turns the implicit time-advance into a linear operator worth analysing.
2. **Factorise** — full eigenvalues, 40 eigenvectors (written out as `.rst` restart
   files), and the complex Schur factor `T` used for all sampling.
3. **Sample** — `config.source.build()` produces points; `sample_sigmin` evaluates
   them. For each `z`, inverse-power iteration via `eigsh` on a `LinearOperator`
   applying `(zI - T)^-* (zI - T)^-1` through LAPACK `trtrs` triangular solves.
   `zI - T` is never materialised densely. Optionally `refine` grows the set.
4. **Emit** — flat `.npy` arrays plus `run_metadata.json`.

`nonmodal plot` reads that output directory back. It needs no HDF5, no operator and
no caches, which is why `run` also saves `pseudo_eigvals.npy` for the overlay.

### Module map

| Module | Responsibility |
|---|---|
| `matrices.py` | `HDF5Matrix`, `assemble_global` (numba) |
| `fields.py` | 7-field block layout, reduction mask, restart output |
| `operator.py` | reduced operator, eigenvalues, Schur factor, caching |
| `sampling.py` | `Bounds`, point sources, `uniform_points`, `mirror_conjugates` |
| `refine.py` | error-driven Delaunay refinement |
| `pseudospectrum.py` | `sample_sigmin`, fork-pool parallelism, contour levels |
| `plotting.py` | tricontour contours, interpolated heatmap |
| `io.py` | run metadata, flat sample IO |
| `config.py` | frozen `RunConfig` / `PlotConfig` |
| `pipeline.py` | `run_pipeline`, `plot_run` |
| `cli.py` | argparse subcommands; the only module touching `Namespace` |

`examples/pseudospectra_intro.ipynb` is the runnable front door. It imports the test
suite's Matrix Market fetcher via `sys.path` (`tests/` is not a package), so it needs
no simulation output. Keep its cells executable as a plain script — no IPython magics —
since that is how they get verified.

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
- **`Im z = 0` is always a sample row** when the region straddles it
  (`_axis_through_zero`). A plain `linspace` steps over zero for even `ny`, and for
  some odd `ny` lands on ~9e-16 instead of 0.0 — which the half-plane filter
  (`imag >= 0`) and `mirror_conjugates` (`imag != 0`) then disagree about, turning
  one row into a duplicated pair ~1e-15 apart. `DEFAULT_GRID_NY` is odd so the axis
  is a lattice point and spacing stays even; an even `ny` still hits the axis, at
  ~9% spacing spread.
- **Half-plane mirroring is decided by the operator, not the grid.** `run_pipeline`
  checks `np.isrealobj(real_jac)`: a real operator has a conjugate-symmetric spectrum,
  so only `Im z >= 0` is evaluated and `mirror_conjugates` recovers the rest.
  `sample_sigmin` never mirrors on its own. `--no-half-plane` forces full sampling.
- **`assemble_global` uses `inmat.shape[0]` for both loop bounds** — square input only.
  It now raises rather than silently mis-assembling.
- **Plots link plotly.js from a CDN** and render blank offline; `--plot-inline-js`
  embeds it. Less pressing now that plotting is a separate command run off the node.
- **`eigsh` gets a fixed `v0`** (`_start_vector`). ARPACK otherwise randomises the start
  vector, which made runs irreproducible and intermittently failed a tolerance on the
  ill-conditioned `bcsstk01`. Do not remove it to "simplify" the call.
- **The CLI is kept 1-1 with the config objects.** Every `RunConfig`/`PlotConfig`
  field has a flag, and every flag reaches a field — there is an audit for this in
  `tests/test_cli.py`. Flags that *cannot* take effect in a given mode are rejected
  rather than ignored (`--refine-rounds` with `--grid-npy`, `--nlevels` with
  `--levels`), because silently dropping a flag on a two-day job is worse than failing.
- **`--timestep` defines the operator**, so it must match the simulation that exported
  the Jacobian. The caches do *not* key on it — change it and delete them by hand.
- **The region is always derived from the spectrum.** There is no flag for explicit
  bounds: the tool is for starting coarse and refining onto features, not sweeping a
  chosen box. Inference needs the spectrum, so sources are only concrete after
  `resolve(eigvals)` — every source has one, returning a `ResolvedSource`, which keeps
  the pipeline from branching on which kind it got.
- **Resolution comes from refinement, not a finer lattice.** `--grid-nx`/`--grid-ny`
  set a coarse initial grid; `--refine-points N` spends up to N *extra* evaluations on
  top of it. N is a ceiling: one point per triangle per round, so a big request needs
  several rounds, and a short run logs the shortfall. There is deliberately no total-points flag and no `near_square` — a total would
  have to be factored back into a shape, and the answer would depend on the arithmetic
  of the number (the old code turned 128 into 8x16 and 127 into 1x127).
- **Adaptive refinement is opt-in** (`--refine-points`, default 0). Measured: 2.05x
  lower interpolation error than uniform at equal budget on `west0479`, roughly a wash
  on nearly-normal matrices. Contour-targeted weighting was tried and measured *worse*
  in every case, so it was dropped — do not re-add it without evidence.

## Testing approach

Tests are analytic or reference-based, never golden-file.

1. **Synthetic** ([test_pseudospectrum.py](tests/test_pseudospectrum.py)) — for a diagonal
   (already Schur-form) operator, `sigma_min(zI - T) = min_i |z - T_ii|` in closed form,
   verified on both the mirrored and full-grid branches.
2. **Real matrices** ([test_matrixmarket.py](tests/test_matrixmarket.py)) — NIST Matrix
   Market matrices spanning exactly normal (`bcsstk01`) to strongly non-normal
   (`west0479`), checked against a dense SVD of `zI - T` (agrees to ~1e-12) and against
   `sigma_min <= dist(z, spectrum)`, a theorem. `bcsstk01` is symmetric hence normal, so
   that bound is an equality there. The non-normal cases additionally assert a *strict*
   gap, so a bug returning eigenvalue distances would not pass.
3. **Refinement earns its place** ([test_refine.py](tests/test_refine.py)) —
   `test_adaptive_beats_uniform_at_equal_budget` requires lower interpolation error than
   uniform sampling against a dense-SVD reference. It uses a Grcar matrix because that
   reference is cheap. If it ever regresses, the feature is not paying for itself.

`tests/matrixmarket.py` handles fetching: pinned sha256 per matrix, disk cache, and
`urllib` needs an explicit `User-Agent` because math.nist.gov 403s the default one.
Missing downloads **skip** unless `NONMODAL_TEST_REQUIRE_DOWNLOADS=1` (CI sets it).
Use `-m "not network"` to exclude them entirely; `NONMODAL_TEST_DATA` relocates the cache.

`tests/conftest.py` forces the Agg backend — `operator.py` imports pyplot at module
scope.
