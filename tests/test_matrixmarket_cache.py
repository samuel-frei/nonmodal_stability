"""Download, caching and checksum behaviour of the Matrix Market helper.

These never touch the network: the transport is stubbed, so they run anywhere.
"""

import gzip
import hashlib
import io
import urllib.error
from typing import Any

import matrixmarket as mm
import numpy as np
import pytest
import scipy.io

TINY_MTX = b"""%%MatrixMarket matrix coordinate real general
2 2 3
1 1 4.0
1 2 1.0
2 2 3.0
"""


def _payload() -> bytes:
  return gzip.compress(TINY_MTX)


def _spec_for(payload: bytes) -> mm.MatrixSpec:
  return mm.MatrixSpec(
    collection='Test', set_name='tiny', name='tiny',
    sha256=hashlib.sha256(payload).hexdigest(),
    order=2, symmetric=False, normality_defect=1.0)


@pytest.fixture
def cache(tmp_path, monkeypatch):
  monkeypatch.setenv('NONMODAL_TEST_DATA', str(tmp_path))
  return tmp_path


def _stub_urlopen(monkeypatch, payload: bytes, calls: list[str]) -> None:
  class _Resp:
    def __enter__(self) -> _Resp:
      return self

    def __exit__(self, *exc: Any) -> None:
      return None

    def read(self) -> bytes:
      return payload

  def fake(request: Any, timeout: float | None = None) -> _Resp:
    calls.append(request.full_url)
    assert request.get_header('User-agent') == mm.USER_AGENT, (
      'math.nist.gov 403s the default urllib agent, so it must always be set')
    return _Resp()

  monkeypatch.setattr(mm.urllib.request, 'urlopen', fake)


def _stub_offline(monkeypatch) -> None:
  def fake(request: Any, timeout: float | None = None) -> Any:
    raise urllib.error.URLError('no route to host')

  monkeypatch.setattr(mm.urllib.request, 'urlopen', fake)


def test_url_follows_matrixmarket_layout() -> None:
  spec = mm.MATRICES['west0479']
  assert spec.url == (
    'https://math.nist.gov/pub/MatrixMarket2/'
    'Harwell-Boeing/chemwest/west0479.mtx.gz')


def test_fetch_downloads_and_caches(cache, monkeypatch) -> None:
  payload = _payload()
  spec = _spec_for(payload)
  calls: list[str] = []
  _stub_urlopen(monkeypatch, payload, calls)

  path = mm.fetch(spec)
  assert path.exists()
  assert path.read_bytes() == payload
  assert len(calls) == 1


def test_cached_file_is_reused_without_network(cache, monkeypatch) -> None:
  payload = _payload()
  spec = _spec_for(payload)
  (cache / spec.filename).write_bytes(payload)
  _stub_offline(monkeypatch)  # any network use would raise

  assert mm.fetch(spec).read_bytes() == payload


def test_corrupted_cache_is_replaced(cache, monkeypatch) -> None:
  payload = _payload()
  spec = _spec_for(payload)
  (cache / spec.filename).write_bytes(b'truncated garbage')
  calls: list[str] = []
  _stub_urlopen(monkeypatch, payload, calls)

  assert mm.fetch(spec).read_bytes() == payload
  assert len(calls) == 1, 'a corrupt cache entry should trigger exactly one re-fetch'


def test_offline_and_uncached_raises_download_unavailable(cache, monkeypatch) -> None:
  spec = _spec_for(_payload())
  _stub_offline(monkeypatch)

  with pytest.raises(mm.DownloadUnavailable):
    mm.fetch(spec)


def test_bad_checksum_is_not_written_to_cache(cache, monkeypatch) -> None:
  payload = _payload()
  spec = _spec_for(payload)
  tampered = mm.MatrixSpec(
    collection=spec.collection, set_name=spec.set_name, name=spec.name,
    sha256='0' * 64, order=2, symmetric=False, normality_defect=1.0)
  _stub_urlopen(monkeypatch, payload, [])

  with pytest.raises(ValueError, match='checksum mismatch'):
    mm.fetch(tampered)
  assert not (cache / tampered.filename).exists()


def test_load_dense_returns_complex_array(cache, monkeypatch) -> None:
  payload = _payload()
  spec = _spec_for(payload)
  _stub_urlopen(monkeypatch, payload, [])

  A = mm.load_dense(spec)
  assert A.shape == (2, 2)
  assert np.iscomplexobj(A)
  expected = scipy.io.mmread(io.BytesIO(TINY_MTX)).toarray()
  np.testing.assert_allclose(A.real, expected)


def test_downloads_required_reads_env(monkeypatch) -> None:
  monkeypatch.delenv('NONMODAL_TEST_REQUIRE_DOWNLOADS', raising=False)
  assert not mm.downloads_required()
  monkeypatch.setenv('NONMODAL_TEST_REQUIRE_DOWNLOADS', '0')
  assert not mm.downloads_required()
  monkeypatch.setenv('NONMODAL_TEST_REQUIRE_DOWNLOADS', '1')
  assert mm.downloads_required()
