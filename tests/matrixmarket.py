"""Fetch test matrices from the NIST Matrix Market.

Downloads are cached on disk and checksum-verified, so the suite runs offline
once warmed and cannot silently absorb a change in upstream content.

Cache location: ``$NONMODAL_TEST_DATA``, else ``tests/_data/`` (gitignored).

Set ``NONMODAL_TEST_REQUIRE_DOWNLOADS=1`` to turn a failed download into an
error instead of a skip. CI sets it; an offline compute node should not, so
that `uv run pytest` there skips these rather than failing.
"""

import gzip
import hashlib
import io
import os
import pathlib
import urllib.error
import urllib.request
from dataclasses import dataclass

import numpy as np
import scipy.io
from numpy.typing import NDArray

BASE_URL = 'https://math.nist.gov/pub/MatrixMarket2'
DOWNLOAD_TIMEOUT_S = 60
#: math.nist.gov answers 403 to urllib's default Python-urllib/x.y agent.
USER_AGENT = 'nonmodal-tests/0.1 (+https://github.com/samuel-frei/nonmodal_stability)'


class DownloadUnavailable(RuntimeError):
  """Raised when a matrix is neither cached nor downloadable."""


@dataclass(frozen=True)
class MatrixSpec:
  """A single Matrix Market matrix and what we expect it to contain."""

  collection: str
  set_name: str
  name: str
  sha256: str
  order: int
  #: Symmetric matrices are normal, so sigma_min(zI - A) == dist(z, spectrum)
  #: exactly. Non-normal ones only satisfy the inequality.
  symmetric: bool
  #: ||AA* - A*A|| / ||A||^2 -- 0 for normal, larger means more non-normal.
  normality_defect: float
  #: Tolerance against a dense-SVD reference. Most agree to ~1e-12; bcsstk01 is
  #: a badly conditioned stiffness matrix, where eigsh's tol=1e-6 dominates.
  svd_rtol: float = 1e-9

  @property
  def url(self) -> str:
    return f'{BASE_URL}/{self.collection}/{self.set_name}/{self.name}.mtx.gz'

  @property
  def filename(self) -> str:
    return f'{self.name}.mtx.gz'


# Chosen to span the normality range: bcsstk01 is exactly normal and gives an
# exact identity to test against; west0479 is the classic strongly non-normal
# pseudospectra example (Trefethen), kept for the slow test.
MATRICES: dict[str, MatrixSpec] = {
  'bcsstk01': MatrixSpec(
    collection='Harwell-Boeing', set_name='bcsstruc1', name='bcsstk01',
    sha256='567560f75b952d9c14c0d193ded5d80370d7ca26fe49f9e67deee55f22e55699',
    order=48, symmetric=True, normality_defect=0.0, svd_rtol=1e-3),
  'olm100': MatrixSpec(
    collection='NEP', set_name='olmstead', name='olm100',
    sha256='9a91e04aa14d54e043f3a60fe26c0ebd27f4bb644a4500886a930e1615218f88',
    order=100, symmetric=False, normality_defect=2.769e-01),
  'gre__115': MatrixSpec(
    collection='Harwell-Boeing', set_name='grenoble', name='gre__115',
    sha256='6b863e4f6f426c7d5ba39827076ec907833b3787ad8ae7c9fbcf871bd4a119d2',
    order=115, symmetric=False, normality_defect=2.743e-02),
  'rw136': MatrixSpec(
    collection='NEP', set_name='mvmrwk', name='rw136',
    sha256='535329fbb282b8520bad1bce5a52e3eec59fd3b42a94fe9997caf715365dd5be',
    order=136, symmetric=False, normality_defect=5.418e-02),
  'west0479': MatrixSpec(
    collection='Harwell-Boeing', set_name='chemwest', name='west0479',
    sha256='aa32caf765f8236338e4899ca8434fad56d524020b60a329671223e6c5d68027',
    order=479, symmetric=False, normality_defect=6.309e-01),
}


def cache_dir() -> pathlib.Path:
  """Directory holding downloaded matrices."""
  override = os.environ.get('NONMODAL_TEST_DATA')
  if override:
    return pathlib.Path(override)
  return pathlib.Path(__file__).resolve().parent / '_data'


def downloads_required() -> bool:
  """True when a failed download should be an error rather than a skip."""
  return os.environ.get('NONMODAL_TEST_REQUIRE_DOWNLOADS', '') not in ('', '0')


def _verify(payload: bytes, spec: MatrixSpec) -> None:
  digest = hashlib.sha256(payload).hexdigest()
  if digest != spec.sha256:
    raise ValueError(
      f'checksum mismatch for {spec.name}: expected {spec.sha256}, got {digest}. '
      'Upstream content changed, or the download was corrupted.')


def fetch(spec: MatrixSpec) -> pathlib.Path:
  """Return a path to the cached .mtx.gz, downloading it if necessary.

  A cached file that fails verification is removed and re-fetched once, so a
  truncated download does not poison the cache permanently.
  """
  target = cache_dir() / spec.filename

  if target.exists():
    try:
      _verify(target.read_bytes(), spec)
      return target
    except ValueError:
      target.unlink()

  request = urllib.request.Request(spec.url, headers={'User-Agent': USER_AGENT})
  try:
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S) as resp:
      payload = resp.read()
  except (urllib.error.URLError, TimeoutError, OSError) as exc:
    raise DownloadUnavailable(f'could not download {spec.url}: {exc}') from exc

  # Verify before writing, so the cache never holds unverified bytes.
  _verify(payload, spec)
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_bytes(payload)
  return target


def load_dense(spec: MatrixSpec) -> NDArray[np.complex128]:
  """Load a Matrix Market matrix as a dense complex array."""
  raw = gzip.decompress(fetch(spec).read_bytes())
  matrix = scipy.io.mmread(io.BytesIO(raw))
  dense = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
  return np.asarray(dense, dtype=np.complex128)
