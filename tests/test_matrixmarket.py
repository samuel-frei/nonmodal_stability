"""Pseudospectrum checks against real matrices from the NIST Matrix Market.

The synthetic tests in test_pseudospectrum.py use diagonal operators, which are
normal and trivially conditioned. These exercise the same code on real,
genuinely non-normal matrices, against two independent references:

* a dense SVD of `zI - T`, which is ground truth for sigma_min; and
* the identity sigma_min(zI - A) = dist(z, spectrum), which holds exactly for
  normal matrices and becomes a strict inequality otherwise. The size of that
  gap is precisely what a pseudospectrum is measuring, so the non-normal cases
  double as a check that the code detects non-normality at all rather than
  reproducing eigenvalue distances.

Downloads are cached and checksum-verified; see matrixmarket.py.
"""

import functools

import numpy as np
import pytest
import scipy.linalg
from matrixmarket import (
  MATRICES,
  DownloadUnavailable,
  MatrixSpec,
  downloads_required,
  load_dense,
)
from numpy.typing import NDArray

from nonmodal import compute_pseudospectrum

pytestmark = pytest.mark.network

NON_NORMAL = ['olm100', 'gre__115', 'rw136', 'west0479']
ALL_MATRICES = ['bcsstk01', *NON_NORMAL]


@functools.cache
def _prepared(name: str) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
  """Return (eigenvalues, Schur factor) for a named matrix.

  Cached because the Schur factorisation is the expensive part and several
  tests reuse it.
  """
  spec = MATRICES[name]
  try:
    A = load_dense(spec)
  except DownloadUnavailable as exc:
    if downloads_required():
      raise
    pytest.skip(f'{name} unavailable offline: {exc}')

  assert A.shape == (spec.order, spec.order)
  eigvals = np.linalg.eigvals(A)
  schur_t = np.asarray(scipy.linalg.schur(A, output='complex')[0], dtype=np.complex128)
  return eigvals, schur_t


def _spectrum_bounds(
  eigvals: NDArray[np.complex128], pad: float = 0.35, symmetric: bool = True
) -> tuple[float, float, float, float]:
  """A padded box around the spectrum.

  Padding keeps sample points off the eigenvalues, where sigma_min collapses
  toward zero and a relative comparison stops being meaningful.
  """
  span = max(
    float(eigvals.real.max() - eigvals.real.min()),
    2.0 * float(np.abs(eigvals.imag).max()),
    1e-12)
  p = pad * span
  imag_extent = float(np.abs(eigvals.imag).max()) + p
  if symmetric:
    return (float(eigvals.real.min()) - p, float(eigvals.real.max()) + p,
            -imag_extent, imag_extent)
  # Asymmetric bounds defeat the half-plane mirroring and force a full grid.
  return (float(eigvals.real.min()) - p, float(eigvals.real.max()) + p,
          -imag_extent, 0.83 * imag_extent)


def _dense_sigmin_reference(
  Z: NDArray[np.complex128], T: NDArray[np.complex128]
) -> NDArray[np.float64]:
  """Ground-truth sigma_min(zI - T) via a dense SVD at every sample point."""
  eye = np.eye(T.shape[0])
  return np.array(
    [[np.linalg.svd(z * eye - T, compute_uv=False)[-1] for z in row] for row in Z])


def _distance_to_spectrum(
  Z: NDArray[np.complex128], eigvals: NDArray[np.complex128]
) -> NDArray[np.float64]:
  return np.min(np.abs(Z[..., None] - eigvals[None, None, :]), axis=-1)


@pytest.mark.parametrize('name', ALL_MATRICES)
def test_sigmin_matches_dense_svd(name: str) -> None:
  """The trtrs/eigsh path must reproduce a dense SVD of zI - T."""
  spec: MatrixSpec = MATRICES[name]
  eigvals, T = _prepared(name)
  rmin, rmax, imin, imax = _spectrum_bounds(eigvals)

  R, C, sigmin = compute_pseudospectrum(
    T, grid_points=16, nprocs=2,
    real_min=rmin, real_max=rmax, imag_min=imin, imag_max=imax)

  reference = _dense_sigmin_reference(R + 1j * C, T)
  np.testing.assert_allclose(sigmin, reference, rtol=spec.svd_rtol)


def test_normal_matrix_sigmin_equals_distance_to_spectrum() -> None:
  """For a normal matrix the pseudospectrum collapses onto eigenvalue disks.

  bcsstk01 is symmetric, hence normal, so sigma_min(zI - A) = dist(z, spectrum)
  holds exactly. This is the closed-form check from the synthetic suite, on a
  real matrix.
  """
  spec = MATRICES['bcsstk01']
  assert spec.symmetric
  eigvals, T = _prepared('bcsstk01')
  rmin, rmax, imin, imax = _spectrum_bounds(eigvals)

  R, C, sigmin = compute_pseudospectrum(
    T, grid_points=25, nprocs=2,
    real_min=rmin, real_max=rmax, imag_min=imin, imag_max=imax)

  distance = _distance_to_spectrum(R + 1j * C, eigvals)
  np.testing.assert_allclose(sigmin, distance, rtol=1e-3)


@pytest.mark.parametrize('name', ALL_MATRICES)
def test_resolvent_lower_bound_holds(name: str) -> None:
  """sigma_min(zI - A) <= dist(z, spectrum) for every matrix.

  Equivalent to ||(zI - A)^-1|| >= 1 / dist(z, spectrum), which is a theorem,
  so a violation beyond round-off means the sampled values are wrong.
  """
  eigvals, T = _prepared(name)
  rmin, rmax, imin, imax = _spectrum_bounds(eigvals)

  R, C, sigmin = compute_pseudospectrum(
    T, grid_points=16, nprocs=2,
    real_min=rmin, real_max=rmax, imag_min=imin, imag_max=imax)

  distance = _distance_to_spectrum(R + 1j * C, eigvals)
  assert np.all(sigmin <= distance * (1.0 + 1e-3))


@pytest.mark.parametrize('name', NON_NORMAL)
def test_nonnormality_produces_a_strict_gap(name: str) -> None:
  """Non-normal matrices must show sigma_min strictly below the eigenvalue distance.

  Without this, a bug that simply returned dist(z, spectrum) would pass every
  other check here.
  """
  eigvals, T = _prepared(name)
  rmin, rmax, imin, imax = _spectrum_bounds(eigvals)

  R, C, sigmin = compute_pseudospectrum(
    T, grid_points=16, nprocs=2,
    real_min=rmin, real_max=rmax, imag_min=imin, imag_max=imax)

  ratio = sigmin / _distance_to_spectrum(R + 1j * C, eigvals)
  assert ratio.min() < 0.9, f'{name} shows no non-normal amplification'


def test_full_grid_path_matches_dense_svd() -> None:
  """Asymmetric imaginary bounds disable mirroring; the full grid must still agree."""
  spec = MATRICES['gre__115']
  eigvals, T = _prepared('gre__115')
  rmin, rmax, imin, imax = _spectrum_bounds(eigvals, symmetric=False)

  R, C, sigmin = compute_pseudospectrum(
    T, grid_points=16, nprocs=2,
    real_min=rmin, real_max=rmax, imag_min=imin, imag_max=imax)

  reference = _dense_sigmin_reference(R + 1j * C, T)
  np.testing.assert_allclose(sigmin, reference, rtol=spec.svd_rtol)
