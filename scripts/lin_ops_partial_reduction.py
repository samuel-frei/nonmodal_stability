import h5py
import scipy
import scipy.sparse as sparse
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

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

def compute_pseudospectrum(A, grid_dim=100, max_iter=100):
  I = sparse.eye(A.shape[0])
  print("computing schur factorization", flush=True)
  # Compute schur factorization
  _, T = scipy.linalg.schur(A, output='complex')
  r = np.linspace(0.99998, 1.0001, grid_dim)
  c = np.linspace(-5e-14, -5e-14, grid_dim)
  R, C =  np.meshgrid(r, c)
  zz = R+C
  sigmin = np.zeros((grid_dim, grid_dim))
  print('Computing pseudospectrum', flush=True)
  for i in range(grid_dim):
    for j in range(grid_dim):
      z = zz[i, j]
      T1 = z*I - T
      T2 = T1.H
      v = np.random.rand(A.shape[0])+ np.random.rand(A.shape[0])*1.0j
      v = v/np.linalg.norm(v)
      sigold = 0
      for _ in range(max_iter):
        tmp = scipy.linalg.solve(T2, v, 'lower triangular')
        v = scipy.linalg.solve(T1, tmp, 'upper triangular')
        sig = np.linalg.norm(v)
        if abs(1-sigold/sig)<0.001:
          break
        sigold=sig
        v = v/sig
      sigmin[i, j] = 1/np.sqrt(sig)
  return R, C, sigmin

if __name__=="__main__":
  # Get jacobian and mass matrices from files
  Mmat = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/mass_mat.h5', '/massmat')
  Jac = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/lin_ops.h5', '/jacobian')
  Mmat = Mmat.csr_rep.todense()
  nr_block=Mmat.shape[0]
  Mmat_big = scipy.linalg.block_diag(Mmat,Mmat,Mmat,Mmat,Mmat,Mmat,Mmat)
  del Mmat  
  dense_Jac = Jac.csr_rep.todense()
  dense_Jac[2*nr_block:3*nr_block, :] = 0
  dense_Jac[:, 2*nr_block:3*nr_block] = 0
  dense_Jac[6*nr_block:, :] = 0
  dense_Jac[:, 6*nr_block:] = 0
  print(Jac.nrg, Jac.ncg, flush=True)
  gl_block=int(Jac.nrg/7)
  print(gl_block)
  jac_gl = make_global_mat(dense_Jac, Jac.nrg, Jac.ncg, Jac.lg)
  jac_gl_reduced = np.block([[jac_gl[:2*gl_block, :2*gl_block], jac_gl[:2*gl_block, 3*gl_block:6*gl_block]], \
                              [jac_gl[3*gl_block:6*gl_block, :2*gl_block], jac_gl[3*gl_block:6*gl_block, 3*gl_block:6*gl_block]]])
  mmat_gl = make_global_mat(Mmat_big, Jac.nrg, Jac.ncg, Jac.lg)
  mmat_gl_reduced = mmat_gl[:5*gl_block, :5*gl_block]
  del Jac, dense_Jac, Mmat_big, jac_gl, mmat_gl
  real_Jac = scipy.linalg.solve(jac_gl_reduced, mmat_gl_reduced)
  w, v = sparse.linalg.eigs(real_Jac, sigma=1, k=10,  which="LM")
  np.save('partial_red/partial_eigs.npy', w)
  plt.scatter(w.real, w.imag)
  plt.savefig('partial_red/partial_spectrum.png')
  plt.clf()
  # R, C, sigmin = compute_pseudospectrum(real_Jac)
  # fig, ax = plt.subplots()
  # ax.scatter(w.real, w.imag)
  # artist = ax.contour(R, C, sigmin)
  # ax.clabel(artist, fontsize=10)
  # ax.set_title('Pseudospectrum')
  # plt.savefig('partial_red/pseudoplot', format='png')