# nonmodal

Nonmodal (pseudospectral) stability analysis of reduced linear MHD operators.

Given exported Jacobian and mass matrices, `nonmodal` reduces the global system
to a subset of physical fields, builds the Schur form of the continuous-time
generator `L` of `du/dt = L u` (eigenvalues are growth rates in 1/s), and samples
the resolvent norm

$$\sigma_{\min}(zI - T)$$

over a region of the complex plane, in the scaled L2 norm of the fields (see
[Norm](#norm)). The $\varepsilon$-pseudospectrum contours
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

## Example

[`examples/pseudospectra_intro.ipynb`](examples/pseudospectra_intro.ipynb) works
one operator through end to end using matrices fetched from the NIST Matrix
Market: inferring a region from the spectrum, coarse sampling then refinement,
contours, the `sigma_min / dist(z, spectrum)` ratio that quantifies how far the
resolvent exceeds what the eigenvalues alone predict, and the pseudomode that
ratio is measuring. On `west0479` the pseudomode beats the best possible
eigenvector by 1.7e6 — the same factor the ratio reports, reached independently.
It needs no simulation output of your own.

```bash
uv run jupyter lab examples/pseudospectra_intro.ipynb   # requires jupyter
```

## Inputs

A run needs three HDF5 files produced by the simulation:

| File | Dataset group | Contents |
|---|---|---|
| `lin_ops.h5` | `/jacobian` | linearised Jacobian |
| `mass_mat.h5` | `/massmat` | mass matrix (one field block) |
| `stiff_mat.h5` | `/stiffmat` | Laplacian stiffness matrix (one field block) |

`stiff_mat.h5` is written only by linear runs of the `nonmodal` branch of our
OpenFUSIONToolkit fork. It is needed to pick out resolved modes; `--all-modes`
runs without it.

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

### Norm

Every resolvent norm, Kreiss constant and pseudomode is measured in the **scaled L2
norm** of the fields,

$$\|x\|_N^2 = \sum_f \int \left(\frac{u_f}{s_f}\right)^2 dA,$$

where `s_f` is the typical size of field `f`, read from the deck beside the export:

| field | `s_f` |
|---|---|
| n | `n0/den_scale` (OFT stores `n/den_scale`) |
| v_x, v_z | Alfven speed at the sheet peak, `psi0/sqrt(mu0 m_p n0)` |
| T | `t0` |
| psi | `psi0*lam_b` |

The integral (the mass matrix) removes the mesh from the norm, so results converge
with resolution; the scales remove the units, so psi counts as much as velocity.
The Euclidean norm of the raw coefficients does neither. In matrix form
`N = R^T R`, and the commands work with `B = R L R^-1`, which has `L`'s eigenvalues.
It is not the energy norm: that weights velocity by the base density and the field
by `|grad psi|^2`, and needs matrices OFT does not export yet.

## Command-line use

There are four subcommands, because their requirements differ. `run` and `kreiss`
need the HDF5 matrices and a machine with cores; `pseudomode` needs the operator but
not the cores; `plot` needs only a finished output directory.

```bash
# On the cluster, from a directory containing the input files.
# Start coarse, then spend the rest of the budget refining onto features.
uv run nonmodal run --grid-nx 25 --grid-ny 25 --refine-points 2600 --nprocs 32

# Anywhere afterwards -- no HDF5, no operator, no caches:
uv run nonmodal plot --output-dir pseudospectrum --min-level 1e-7 --nlevels 16
```

```bash
# The mode behind a point you name, as an OFT restart file:
uv run nonmodal pseudomode --at 5e5-2.4e4j
uv run nonmodal pseudomode --at 5e5-2.4e4j --at -1e6+3e5j --phases 8
```

```bash
# A rigorous lower bound on the Kreiss constant, and so on transient growth:
uv run nonmodal kreiss --nprocs 32
```

`python -m nonmodal` works identically. `cases/harris_linear_20x6z/ps1.sbatch` runs
both steps as one SLURM job: submit it from the directory holding the exports, and
set `NONMODAL_REPO` if this checkout is not the one it should use.

### Kreiss constant

`K(L) = sup_{Re z > 0} Re(z) ||(zI - L)^-1||` brackets the transient growth the
non-normality permits: `K <= sup_t ||e^{tL}|| <= e n K` (Kreiss 1962; the sharp
factor `e n` is Spijker 1991). Certified global algorithms (Mitchell 2020) cost
`O(n^4)` or more, out of reach at `n = 10500`, so `kreiss` computes a lower bound:

1. a log-spaced scan of `x / sigma_min(x)` along the positive real axis, refined
   with a bounded 1-D optimiser. For a real operator the resolvent norm is
   symmetric about the real axis, which is therefore where the supremum is
   expected -- an assumption the next two steps test rather than trust;
2. a check that the real-axis maximiser is a maximum, not a saddle, in `Im z`;
3. a coarse log grid over the right half-plane; if it beats the real axis, that
   point is polished locally, reported instead, and flagged.

Every evaluated point is a true lower bound on `K`, so a wrong assumption weakens
the bound rather than corrupting it. `kreiss.json` records `K`, `z*`, both theorem
bounds, the two checks, the real-axis scan, and the physical scales. A stable
normal operator gives `K -> 1` only as `x -> infinity`; the result then reports the
maximum at the scan's upper end. An unstable operator has `K = infinity` and is
rejected.

### Resolved modes

On a finite-element mesh the stiff end of the spectrum belongs to the grid, and
that is where the resolvent norm of the whole operator peaks: on the 10x3z Harris
mesh the Kreiss maximiser sat at the spectral radius, with one oscillation per
element. So by default `run`, `pseudomode` and `kreiss` analyse only the modes
the mesh resolves.

Each eigenmode gets an effective wavenumber per field,
`k_f^2 = x_f K x_f / x_f M x_f`, from the exported Laplacian `K` and scalar mass
`M`, and the mesh's largest is `k_max^2 = max eig(M^-1 K)`. A mode is kept when
`k_f / k_max < --resolution-cutoff` (default 0.1) in every field it occupies -- the
worst field decides, however little of the mode's energy it carries. The Schur
form is then reordered so its leading block is `L` restricted exactly to the kept
modes' span, in the same norm. `--all-modes` analyses all of `L`.

The per-mode wavenumbers are cached beside the Schur factor
(`mode_resolution_l2s.npz`), so trying another cutoff costs only the reorder.
`run_metadata.json`, `pseudomodes.json` and `kreiss.json` record the cutoff,
`k_max` per field, and the kept eigenvalues under `subspace`.

### Pseudomodes

The pseudomode at `z` is the right singular vector of `zI - A` belonging to
`sigma_min`, so it satisfies `||(A - z I) v|| = sigma_min`: an approximate eigenvector
with a known residual. Where the operator is strongly non-normal, `sigma_min` is tiny
far from any eigenvalue, and `v` is a direction the operator very nearly leaves
invariant even though `z` is nowhere near the spectrum.

Extraction runs the same inverse iteration the sampler runs, at the point you name.
`sigma_min` and its eigenvector come out of one solve, and that eigenvector *is* the
pseudomode — sampling computes it at every point and discards it. Keeping it costs
nothing beyond the value, but the solve is still done: sampled vectors are not stored, so
a mode is a fresh iteration even at a point the run already visited. Multiplying by the
Schur vectors then returns it to the physical basis.

Modes are written to `<output-dir>/pseudomodes/` as `.rst` restart files, the same format
as the eigenvectors, for viewing in OFT. `--phases N` sweeps the mode's phase across `N`
files; since each file records its index as `t`, the directory reads back as a time
series and a travelling mode animates. The default `--phases 1` writes the single phase
carrying the most amplitude.

`--at` takes the point directly and may be repeated; several points share one load of the
operator. Choosing *which* point is deliberately left to you — nothing here searches the
plane. `pseudomodes.json` records each `z`, its `sigma_min`, and the residual.

### Sampling

**The region always comes from the spectrum.** There is no way to hand-pick a
rectangle: this tool is built to start at low resolution and zoom in on
features, not to sweep a chosen box at high resolution. `--bounds-pad`
(default 0.3 of the spectral span) sets how much room to leave around the
eigenvalues, and the box it picks is logged.

**Resolution is asked for as dimensions, not a total.** `--grid-nx` and
`--grid-ny` give the initial lattice (default 25x25). Keep it coarse. You can
also supply a precomputed flat complex array with `--grid-npy points.npy`;
samples are unstructured throughout, so no shape is needed.

`Im z = 0` is always one of the sampled rows: a real operator's pseudospectrum
is symmetric about the real axis and its contours pinch there, so it is the last
line you want to step over. `--grid-ny` is odd by default so the axis is a
lattice point with even spacing either side; an even value still lands on the
axis, at slightly uneven spacing.

**`--refine-points N` is how you reach resolution.** It spends *up to* `N`
further evaluations on top of the initial grid, inserting them where a linear
interpolant of `log10(sigma_min)` is worst — which is exactly the error the
contour plot shows. Against a dense-SVD reference at equal total cost this
roughly halves interpolation error on a strongly non-normal operator (2.05x on
`west0479`) and is about a wash on nearly-normal ones, so it is off by default.
`--refine-rounds` (default 4) sets how many passes to spread those points over.

`N` is a ceiling rather than a promise: a round inserts at most one point per
triangle, and an *n*-point set triangulates into roughly 2*n* triangles, so one
round can only about triple it. A run that finishes short says so in the log.
Four rounds comfortably covers the 25x25 → +2600 case.

If the reduced operator is real — it is — its spectrum is conjugate-symmetric,
so only the upper half-plane is evaluated and the rest follows by conjugation.
`--no-half-plane` disables that.

### Outputs

`run` writes to `--output-dir` (default `pseudospectrum/`):

- `pseudo_z.npy`, `pseudo_sigmin.npy` — the sampled points and their values,
  both flat
- `pseudo_eigvals.npy` — the spectrum, so `plot` can draw its overlay
  standalone
- `run_metadata.json` — sampling strategy, bounds, worker counts, input paths, the
  norm and its field scales, and the physical scales
- `eigvecs_plot/xmhd2d_*.rst` — the rightmost eigenvectors as restart files, with
  `eigvecs_plot/eigenmodes.json` naming which eigenvalue each file holds. Selection
  is by real part alone, so a conjugate pair contributes two files showing the same
  mode.

`pseudomode` writes `pseudomodes/xmhd2d_*.rst` alongside `pseudomodes.json`.

`plot` adds `*_heatmap.html` and `*_contours.html`. Contours are drawn directly
from a triangulation of the samples, so they carry no interpolation error; the
heatmap interpolates `log10(sigma_min)` onto a regular mesh (`--plot-mesh`)
because a raster needs one.

> **Note:** plots link plotly.js from a CDN and so render blank without network
> access. Pass `--plot-inline-js` to embed it instead (~5 MB per file).

Runs are bitwise reproducible: the iterative solver is given a fixed starting
vector rather than ARPACK's random one.

## Caching — read this before trusting a result

Four expensive intermediates are cached under `--cache-dir` (default `.`):

```
generator.npy  generator_schur_l2s.npy  generator_schurvecs_l2s.npy  mode_resolution_l2s.npz
```

`mode_resolution_l2s.npz` is the one exception to filename-only keys: it stores the
eigenvalues it was computed for and is ignored unless they equal `diag(T)` exactly.
It does not notice a changed `stiff_mat.h5`, so delete it after re-exporting.

`generator.npy` is `L` itself, shared by every norm; the `_l2s` files factorise it in
the scaled L2 norm. Untagged `generator_schur*.npy` files are factors in the plain
l2 coefficient norm, which only the library writes, and the commands never read them.
`generator_schurvecs_l2s.npy` holds the unitary `Z` of `B = Z T Z*`. Sampling never needs
it, but eigenvectors and pseudomodes do, and `Z` cannot be recovered from a cached `T` —
so a cache holding only `T` has to redo the factorisation once. It says so before
starting.

**Caches are keyed by filename alone.** They are *not* invalidated when the
input matrices or `KEPT_BLOCK_IDS` change. Changing either while stale caches are
present will silently produce results for the *previous* problem.

Delete them by hand after any such change:

```bash
rm -f generator*.npy mode_resolution*.npz
```

Older runs left `real_jacobian.npy` and `full_reduced_*.npy`. Those hold the
backward-Euler operator `(J^-1 M - I)/dt`, not `L`, and are never read; the run log
says so when it finds them.

Every cache hit logs its absolute path and mtime so you can spot a stale reuse
in the run log.

## Library use

The primitive is `sample_sigmin`, which evaluates a flat array of complex
points. Everything else — uniform grids, refinement, mirroring — is a question
of which points you hand it.

```python
import numpy as np
from nonmodal import Bounds, sample_sigmin, uniform_points

points = uniform_points(Bounds(-9e6, 5e5, -3e6, 3e6), nx=64, ny=64)
sigmin = sample_sigmin(points, schur_t, nprocs=16)
```

`compute_pseudospectrum` wraps that for the rectangular case, returning
`(R, C, sigmin)` meshes ready for contouring. The pipeline does not use it — it
samples flat and refines — but it is convenient when you genuinely want a fixed
lattice:

```python
from nonmodal import Bounds, compute_pseudospectrum

R, C, sigmin = compute_pseudospectrum(
    schur_t, Bounds(-9e6, 5e5, -3e6, 3e6), nx=32, ny=32, nprocs=16)
```

`Bounds.around_spectrum(eigvals, pad=0.3)` builds the same region `run` uses.

A pseudomode needs both halves of the Schur factorisation, since `sigma_min` is
invariant under `L = Z T Z*` but its singular vectors are not:

```python
from nonmodal import load_or_compute_schur_vectors, pseudomode_at

schur_t, schur_z = load_or_compute_schur_vectors(generator, cache_dir='.')
mode = pseudomode_at(schur_t, schur_z, 5e5 - 2.4e4j)
print(mode.sigma_min, mode.residual)   # equal, and that is what identifies it
```

`sigmin_with_mode` is the same thing without the change of basis, returning the
vector in the Schur basis.

Every module opens with a one-line index of what it exports; `help(nonmodal.io)`
and friends are the fastest way in. The package ships `py.typed`.

Two behaviours worth knowing:

- **Half-plane mirroring.** `run` decides from the operator itself
  (`np.isrealobj`) whether the spectrum is conjugate-symmetric, and if so
  samples only `Im z >= 0`, recovering the rest with `mirror_conjugates`.
  `sample_sigmin` never does this on your behalf — mirroring is something you
  opt into by choosing the points.
- **BLAS threads.** Importing `nonmodal` pins `OMP/MKL/OPENBLAS_NUM_THREADS` to
  `1`, because the sampling pool would otherwise oversubscribe every core. It
  uses `setdefault`, so if you want more threads for your own dense linear
  algebra, set them *before* importing.

## Development

```bash
uv run ruff check . && uv run mypy && uv run pytest
```

### Test matrices

Part of the suite validates the sampler against real matrices downloaded from
the [NIST Matrix Market](https://math.nist.gov/MatrixMarket/) — spanning exactly
normal (`bcsstk01`) to strongly non-normal (`west0479`, the classic
pseudospectra example). They are checked against a dense SVD of `zI - T`, and
against the identity `sigma_min(zI - A) = dist(z, spectrum)` that holds exactly
for normal matrices and strictly fails otherwise.

Downloads are checksum-verified and cached, so the suite runs offline once
warmed. On a machine that has never fetched them and has no network, these tests
**skip** rather than fail.

```bash
uv run pytest -m "not network"      # never touch the network
NONMODAL_TEST_DATA=/shared/mm uv run pytest   # cache elsewhere
NONMODAL_TEST_REQUIRE_DOWNLOADS=1 uv run pytest   # treat a failed download as an error (CI does this)
```

## Repository layout

```
src/nonmodal/   the library
examples/       the introductory notebook
cases/          OFT input decks and submission scripts
notebooks/      exploratory analysis
tests/
```

`cases/` holds the simulation inputs that generate the `.h5` files this package
consumes; `run_script.sh` in each case invokes the solver itself, not this code.
