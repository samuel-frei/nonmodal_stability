import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import h5py
import scipy
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import multiprocessing
from multiprocessing import shared_memory

class Matrix:
  def __init__(self, filename, mat_name):
    self.file = h5py.File(filename, 'r')
    self.mat_name = mat_name
    self.nr = self.file[f'{mat_name}/nr'][0]
    self.nc = self.file[f'{mat_name}/nc'][0]
    self.nrg = self.file[f'{mat_name}/nrg'][0]
    self.ncg = self.file[f'{mat_name}/ncg'][0]
    self.lc = np.array(self.file[f'{mat_name}/lc'])-1
    self.lg = np.array(self.file[f'{mat_name}/lg'])-1
    self.kr = np.array(self.file[f'{mat_name}/kr'])-1
    self.M = np.array(self.file[f'{mat_name}/M'])
    self.csr_rep = sparse.csr_array((self.M, self.lc, self.kr))
  def spy(self):
    plt.spy(self.csr_rep, markersize = 1, precision='present')
    plt.savefig(f'{self.mat_name}_spy.png')

@njit(parallel=True)
def make_global_mat(inmat, nrg=None, ncg=None, lg=None):
  outmat = np.zeros((nrg, ncg))
  for i in prange(inmat.shape[0]):
    for j in range(inmat.shape[0]):
      ik = lg[i]
      jl = lg[j]
      outmat[ik, jl] += inmat[i, j]
  return outmat

_worker_T = None
_worker_I = None
_worker_trtrs = None
_shm = None

def _init_worker(shm_name, T_shape, T_dtype):
    global _worker_T, _worker_I, _worker_trtrs, _shm
    _shm = shared_memory.SharedMemory(name=shm_name)
    _worker_T = np.ndarray(T_shape, dtype=T_dtype, buffer=_shm.buf)
    _worker_I = np.eye(T_shape[0], dtype=T_dtype)
    # Resolve the raw LAPACK triangular solve once, skip Python overhead per call
    _worker_trtrs, = scipy.linalg.get_lapack_funcs(('trtrs',), (_worker_T,))

def _compute_sig_point(args):
    i, j, z = args
    trtrs = _worker_trtrs
    T = _worker_T
    I = _worker_I
    T1 = z*I - T

    def _matvec(q):
      tmp, _ = trtrs(T1, q.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
      result, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
      return result.ravel()
    
    op = scipy.sparse.linalg.LinearOperator(T1.shape, matvec=_matvec, dtype=np.complex128)
    vals, _ = sparse.linalg.eigsh(op, k=1, which='LM')
    sig_min=vals[0]
    return i, j, 1/np.sqrt(sig_min)

def compute_pseudospectrum(imat, grid_dim=10, nprocs=8):
  T = imat

  # Create a POSIX shared memory block — workers attach with zero-copy
  shm = shared_memory.SharedMemory(create=True, size=T.nbytes)
  T_shm = np.ndarray(T.shape, dtype=T.dtype, buffer=shm.buf)
  np.copyto(T_shm, T)
  
  r = np.linspace(0.998, 1.01, grid_dim)
  c = np.linspace(-5e-3, 5e-3, grid_dim)
  R, C =  np.meshgrid(r, c)
  zz = R + 1j * C
  sigmin = np.zeros((grid_dim, grid_dim))
  print("computing pseudospectrum", flush=True)
  work_items = [(i, j, zz[i, j]) for i in range(grid_dim) for j in range(grid_dim)]
  ctx = multiprocessing.get_context('forkserver')
  with ctx.Pool(
     processes=nprocs,
     initializer=_init_worker, 
     initargs=(shm.name, T.shape, T.dtype)) as pool:
    for i, j, val in pool.imap_unordered(_compute_sig_point, work_items, chunksize=16):
        sigmin[i, j] = val
  shm.close()
  shm.unlink()
  return R, C, sigmin

if __name__=="__main__":
  # Get jacobian and mass matrices from files
  Mmat = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/mass_mat.h5', '/massmat')
  Jac = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/lin_ops.h5', '/jacobian')
  Mmat = Mmat.csr_rep.todense()
  nr_block=Mmat.shape[0]
  Mmat_big = scipy.linalg.block_diag(Mmat,Mmat,Mmat,Mmat,Mmat,Mmat,Mmat)
  del Mmat  
  # zero irrelevant blocks of the jacobian and plot its sparsity pattern
  dense_Jac = Jac.csr_rep.todense()
  dense_Jac[:nr_block, :] = 0
  dense_Jac[:, :nr_block] = 0
  dense_Jac[2*nr_block:3*nr_block, :] = 0
  dense_Jac[:, 2*nr_block:3*nr_block] = 0
  dense_Jac[4*nr_block:5*nr_block, :] = 0
  dense_Jac[:, 4*nr_block:5*nr_block] = 0
  dense_Jac[6*nr_block:, :] = 0
  dense_Jac[:, 6*nr_block:] = 0
  # plt.spy(dense_Jac, markersize=0.5)
  # plt.savefig('reduced/dense_jac_red.png')
  # plt.clf()
  print(Jac.nrg, Jac.ncg, flush=True)
  # assemble global matrices and reduce them to the relevant blocks
  gl_block=int(Jac.nrg/7)
  jac_gl = make_global_mat(dense_Jac, Jac.nrg, Jac.ncg, Jac.lg)
  jac_gl_reduced = np.block([[jac_gl[gl_block:2*gl_block, gl_block:2*gl_block], jac_gl[gl_block:2*gl_block, 3*gl_block:4*gl_block], jac_gl[gl_block:2*gl_block, 5*gl_block:6*gl_block]], \
                              [jac_gl[3*gl_block:4*gl_block, gl_block:2*gl_block], jac_gl[3*gl_block:4*gl_block, 3*gl_block:4*gl_block], jac_gl[3*gl_block:4*gl_block, 5*gl_block:6*gl_block]], \
                              [jac_gl[5*gl_block:6*gl_block, gl_block:2*gl_block], jac_gl[5*gl_block:6*gl_block, 3*gl_block:4*gl_block], jac_gl[5*gl_block:6*gl_block, 5*gl_block:6*gl_block]]])
  # plt.spy(jac_gl_reduced, markersize=0.5)
  # plt.savefig('reduced/jac_gl_red.png')
  # plt.clf()
  mmat_gl = make_global_mat(Mmat_big, Jac.nrg, Jac.ncg, Jac.lg)
  mmat_gl_reduced = mmat_gl[:3*gl_block, :3*gl_block]
  # plt.spy(mmat_gl_reduced, markersize=0.5)
  # plt.savefig('reduced/mmat_gl_red.png')
  # plt.clf()
  del Jac, dense_Jac, Mmat_big, jac_gl, mmat_gl
  # compute real jacobian
  real_Jac = scipy.linalg.solve(jac_gl_reduced, mmat_gl_reduced )
  del jac_gl_reduced, mmat_gl_reduced
  # plt.spy(real_Jac, markersize=0.5)
  # plt.savefig('reduced/real_jac_red.png') 
  # plt.clf()
  # Compute eigenvalues
  w, v = sparse.linalg.eigs(real_Jac, k=15,  which="LM")
  del v
  np.save('reduced/full_reduced_eigs.npy', w)
  print(w, flush=True)
  plt.scatter(w.real, w.imag)
  plt.savefig('reduced/full_reduced_spectrum.png')
  # Compute schur factorization
  # if full_reduced_schur.npy already exists, load it, otherwise compute it and save it to avoid recomputation in the future
  # try:
  #   T = np.load('reduced/full_reduced_schur.npy')
  # except FileNotFoundError:
  #   print("computing schur factorization", flush=True)
  #   _, T = scipy.linalg.schur(real_Jac, output='complex')
  #   np.save('reduced/full_reduced_schur.npy', T)
  # del real_Jac
  # R, C, sigmin = compute_pseudospectrum(T, grid_dim=16, nprocs=12)
  # fig, ax = plt.subplots()
  # ax.scatter(w.real, w.imag)
  # artist = ax.contour(R, C, sigmin)
  # ax.clabel(artist, fontsize=10)
  # ax.set_title('Pseudospectrum')
  # plt.savefig('reduced/pseudoplot.png', format='png')