# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`nonmodal` computes **nonmodal (pseudospectral) stability diagnostics** for reduced
linear MHD operators. It loads exported Jacobian and mass matrices, reduces the
global system to a subset of physical fields, builds the Schur form of the
continuous-time generator `L` (`du/dt = L u`, eigenvalues in 1/s), and samples
`sigma_min(zI - T)` across the complex plane.

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

- `research/nonmodal_archive/recon_data/scripts/lin_ops_full_reduction.py` — equals upstream commit
  `0d0ae3a`, three commits behind, and still contains adaptive-grid code that upstream
  deliberately deleted in `304ff7f` ("remove naive mesh refinement").
- `research/nonmodal_archive/recon_data/psc_data/scripts/lin_ops_full_reduction.py` — was *ahead* of
  upstream; its external-grid work has since been merged here (commit `ef52242`).

This repo is ground truth. Do not treat those copies as a compatibility constraint.

## Where the real data lives

Nothing in this repo; the exports sit on feynman under
`/mnt/homes_global/swf2112/research/nonmodal_archive/recon_data/psc_data/` (archived
Oct 2026; old sbatch scripts in there still use the pre-archive absolute paths).

- `harris_linear_20x6z/{pseudo_marginal,pseudo_hires,pseudo_hhres,pseudo_hhhres,
  margin_finding}/` — the production runs, Apr 2026. Reduced operator is
  **10500x10500** (an operator cache is 882 MB, its Schur factor 1.76 GB).
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
   `load_or_compute_generator` assembles global matrices and forms the generator
   `L = spsolve(M, M - J) / dt` (`generator_from_matrices`). OFT's backward-Euler
   Jacobian is exactly `J = M + dt*K` (checked in source and to 1e-16 numerically),
   so this is `-M^-1 K`, independent of `dt`, and its eigenvalues are physical growth
   rates in 1/s. The pipeline used to analyse `(J^-1 M - I)/dt` instead, which is
   `L(I - dt L)^-1`: right for slow modes, but it squashes the stiff end of the
   spectrum against `-1/dt` (`||A|| ~ 9.4e6` vs `1/dt = 1e7` on the production mesh),
   and on a test case it under-reported the resolvent by 4x at `z = 1e6 i`.
   The exported `/massmat` is the **scalar** Lagrange mass matrix (`nrg` = 2220,
   one field block) which `mass_blocks` replicates across the seven fields —
   scaling the temperature block by `1/(gamma-1)`, exactly as `mfun_apply` in
   OFT's `xmhd_2d.F90` does, and the psi block by `1/eta` for exports from
   upstream OFT (see the psi landmine). `gamma` and `eta` are read from the deck
   beside the export; nothing else in the deck reaches the operator.
2. **Factorise** — one complex Schur factorisation `L = Z T Z*` serves everything
   downstream. The spectrum *is* `diag(T)`, so there is no separate `eigvals` call;
   the 40 rightmost eigenvectors come out of the same `T` by back-substitution
   (`eigenvectors_from_schur`) mapped back through `Z`, written as `.rst` restarts
   with an `eigenmodes.json` naming which eigenvalue each file holds; and `T` alone
   drives all sampling. `Z` is dropped before step 3.
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
| `adapters.py` | OFT-format interop: the input deck (`oft.in` / `oft_surf.in`), `gamma`, `eta`, `dt`, physical scales |
| `matrices.py` | `HDF5Matrix`, `assemble_global` (numba) |
| `fields.py` | 7-field block layout, reduction mask, restart output |
| `operator.py` | reduced generator `L`, Schur factor, eigenvalues and eigenvectors from it, caching |
| `sampling.py` | `Bounds`, point sources, `uniform_points`, `mirror_conjugates` |
| `refine.py` | error-driven Delaunay refinement |
| `pseudospectrum.py` | `sample_sigmin`, fork-pool parallelism, contour levels |
| `plotting.py` | tricontour contours, interpolated heatmap |
| `pseudomode.py` | pseudomode extraction at an explicitly given point |
| `io.py` | run metadata, flat sample IO |
| `config.py` | frozen `RunConfig` / `PlotConfig` / `PseudomodeConfig` |
| `pipeline.py` | `run_pipeline`, `plot_run`, `pseudomode_run` |
| `cli.py` | argparse subcommands; the only module touching `Namespace` |

`examples/pseudospectra_intro.ipynb` is the runnable front door. It imports the test
suite's Matrix Market fetcher via `sys.path` (`tests/` is not a package), so it needs
no simulation output. Keep its cells executable as a plain script — no IPython magics —
since that is how they get verified.

### Naming rule

**Do not use `oft` / `OFT` in our identifiers.** Those names are reserved for genuine
OpenFUSIONToolkit interop, which is what `adapters.py` now is, so that a local module
never sits confusingly beside the real package. This is why the HDF5 reader is
`HDF5Matrix` and the block layout lives in `fields.py` rather than a vendor-named
module. `adapters.py` reads OFT's own file formats, so `oft.in` and `oft_surf.in`
appear there as filenames — but its functions are still named for what they do, not for the vendor.

## Landmines

- **Every cache under the old names holds a different operator.** `real_jacobian.npy`
  and `full_reduced_schur*.npy` are the backward-Euler `A`, not `L`, and those
  predating Aug 2026 also lack the temperature mass factor `1/(gamma-1)`. The
  generator caches are named `generator*.npy` so they can never be mistaken for them;
  `load_or_compute_generator` logs when legacy files sit in the cache dir unused. Any
  result derived from the old caches, including the Apr 2026 production plots, is
  for `A` and predates the fix.
- **`gamma` and `eta` come from the data, not flags.** `load_or_compute_generator`
  finds the deck beside the Jacobian, walking up to three parents (run directories
  sit under a case root that holds the canonical deck; `margin_finding/` and
  `pseudo_marginal/` rely on this). Two deck generations exist: upstream's Harris
  driver reads `oft.in` with `&harris_sheet_options`; the older linear driver read
  `oft_surf.in` with `&xmhd_options`. Either is accepted, but both names at one
  level, both groups in one deck, or no deck at all raises rather than guessing,
  because a wrong value silently yields a wrong operator. There is deliberately no
  `--gamma` or `--eta`: they are properties of the export, like the mesh.
- **The psi mass scale is read off the Jacobian, not the deck.** Upstream OFT
  (commit `5da9985`, 2026-08-31) divides the psi rows of both `mfun_apply` and the
  Jacobian by `eta(2)`; older builds, including every Apr 2026 export, do not. The
  mass block must match or psi dynamics are wrong by a factor `eta` per step.
  `psi_mass_scale` projects the Jacobian's psi-psi block onto the mass matrix:
  that ratio is ~1 for old exports and ~`1/eta` (~1070 here) for new ones, and
  anything near neither raises. The deck name cannot decide this: a fork build
  from Aug 2026 already read `&harris_sheet_options` with the unscaled code. A
  scalar `eta` is only valid for a single-region mesh; OFT stores it per region.
- **Thread pinning must stay at the top of `__init__.py`**, above every import.
  `os.environ.setdefault` for `OMP/MKL/OPENBLAS_NUM_THREADS` only takes effect before
  NumPy is first imported. Moving or reordering it silently degrades many-core runs
  instead of erroring. It uses `setdefault` so a caller who sets threads *before*
  importing (the notebook does, wanting 16) keeps their choice.
- **Fork worker globals.** `_worker_T` / `_worker_trtrs` in `pseudospectrum.py` are set
  in the parent and inherited via `fork`. They must stay in the same module as
  `_compute_sig_point`, and be set on that module object, not rebound locally.
- **Caches are keyed by filename only.** `generator.npy`, `generator_schur*.npy` under
  `--cache-dir` are reused blindly; they are *not* invalidated when inputs or
  `KEPT_BLOCK_IDS` change. Delete them by hand. Cache hits log path and mtime so stale
  reuse is visible in the log.
- **`Z` cannot be recovered from a cached `T`.** `generator_schurvecs.npy` is written
  by `_compute_schur` alongside `generator_schur.npy`. A cache dir holding only `T`
  makes `load_or_compute_schur_vectors` redo the whole `O(n^3)` factorisation on what
  looks like a cache hit, and it says so first. `run` needs `Z`, because that is where
  the eigenvectors come from. `load_or_compute_schur` stays vector-free and is still
  the right entry point for anything that only samples, but nothing in `run` calls it.
- **Eigenvector selection is by real part alone, with no conjugate-pair handling.**
  A real operator's pairs give identical `Re(v)`, so five of the rightmost 40 restarts
  on the production run are redundant twins (verified `|cos| = 1`, sometimes sign-
  flipped since `aligned_phase` resolves θ only mod π). Deduplicating them was tried and
  removed: deciding which entries pair up means judging which roundoff counts as zero,
  and a complex Schur form gives numerically-real eigenvalues an imaginary part of ~5e-9
  that lands on either side of the axis. An `imag >= 0` filter silently dropped
  `-27.9045`, the second-rightmost mode. Sorting cannot lose a mode; that judgement can.
  `eigenmodes.json` records the eigenvalue per file, so the twins are visible.
- **The pseudomode is the eigenvector the sampler discards.**
  `_shifted_resolvent_operator` is `((zI-T)*(zI-T))^-1` — the Gram-matrix formulation of
  Wright & Trefethen (2001) §2, "solved in two stages, each using triangular system
  solves", which is `M^-1 M^-H`. Its eigenvector is therefore the **right** singular
  vector, i.e. the pseudomode; that follows from `M*M` and is not an artefact of how the
  stages are ordered. Swapping them computes `(M M*)^-1`, whose eigenvalues are identical
  — so `sigma_min` is unaffected and no sampling test would catch it — but whose
  eigenvector is the left vector.
- **`Im z = 0` is always a sample row** when the region straddles it
  (`_axis_through_zero`). A plain `linspace` steps over zero for even `ny`, and for
  some odd `ny` lands on ~9e-16 instead of 0.0 — which the half-plane filter
  (`imag >= 0`) and `mirror_conjugates` (`imag != 0`) then disagree about, turning
  one row into a duplicated pair ~1e-15 apart. `DEFAULT_GRID_NY` is odd so the axis
  is a lattice point and spacing stays even; an even `ny` still hits the axis, at
  ~9% spacing spread.
- **Half-plane mirroring is decided by the operator, not the grid.** `run_pipeline`
  checks `np.isrealobj(generator)`: a real operator has a conjugate-symmetric spectrum,
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
- **`--timestep` must equal the deck's `dt`.** It no longer defines the operator —
  `L` is `dt`-free — but it is how `dt` is divided back out of `J = M + dt*K`, so any
  other value gives a wrong `L`. `check_timestep` raises on a mismatch rather than
  trusting the flag.
- **`eta` and `nu` changed units between drivers.** The old linear driver
  (`&xmhd_options` decks, the Apr 2026 exports) read `eta` as resistivity in Ohm m
  (dividing by `mu0`) and `nu` as dynamic viscosity in Pa s (dividing by `m_i n`).
  Upstream OFT (`&harris_sheet_options`) reads both as diffusivities in m^2/s. The Apr
  2026 Jacobian confirms the old reading: its psi diagonal implies a mesh stiffness of
  731 m^-2 with `eta/mu0`, and an impossible 5.8e8 m^-2 without. So **an old deck run
  through the new driver is ~1e6 x less resistive and ~1e10 x less viscous** than
  intended; convert `eta -> eta/mu0` (742.6) and `nu -> nu/(m_p n0)` (7424) first. Even
  converted, the viscosity models differ: old held dynamic viscosity constant, new
  holds kinematic constant (~6x apart in the low-density wings).
  `read_physical_scales` picks the conversion by the deck's group and records which
  in `run_metadata.json`, so tau_A, S and Pm are right for both generations.
- **Growth rates are in 1/s; quote them against `tau_A`.** `run_metadata.json` and
  `pseudomodes.json` carry `physical_scales`: `tau_A = lam_b / v_A` with
  `v_A = psi0 / sqrt(mu0 m_p n0)`, `S = lam_b v_A / D`, `Pm = nu / D`. The reference
  length (sheet half-width) and density (sheet peak `n0`, not the `0.2 n0` background)
  are conventions: changing the density moves `tau_A` and `S` by sqrt(5), so state
  them with any number. For 20x6z: `tau_A = 3.10 us`, `S = 160`, `Pm = 10`.
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
