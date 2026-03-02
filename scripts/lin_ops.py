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
  c = np.linspace(-5e-14, 5e-14, grid_dim)
  R, C =  np.meshgrid(r, c)
  zz = R+C
  sigmin = np.zeros((grid_dim, grid_dim))
  trtrs, = scipy.linalg.get_lapack_funcs(('trtrs',), (T,))
  print('Computing pseudospectrum', flush=True)
  for i in range(grid_dim):
    for j in range(grid_dim):
      z = zz[i, j]
      T1 = z*I - T
      v = np.random.rand(A.shape[0])+ np.random.rand(A.shape[0])*1.0j
      v = v/np.linalg.norm(v)
      sigold = 0
      for _ in range(max_iter):
        tmp, _ = trtrs(T1, v.reshape(-1, 1), lower=0, trans=2, unitdiag=0)
        v, _ = trtrs(T1, tmp, lower=0, trans=0, unitdiag=0)
        sig = np.linalg.norm(v)
        if abs(1-sigold/sig)<0.001:
          break
        sigold=sig
        v = v/sig
      sigmin[i, j] = 1/np.sqrt(sig)
  return R, C, sigmin

if __name__=="__main__":
  # try to load real jacobian from file, if it doesn't exist, compute it and save to file
  try:
    real_Jac = np.load('real_jacobian.npy')
  except FileNotFoundError:
    # Get jacobian and mass matrices from files
    Mmat = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/mass_mat.h5', '/massmat')
    Jac = Matrix('/ocean/projects/phy240045p/freiberg/pseudospectra/harris_linear/lin_ops.h5', '/jacobian')
    Mmat = Mmat.csr_rep.todense()
    nr_block=Mmat.shape[0]
    Mmat_big = scipy.linalg.block_diag(Mmat,Mmat,Mmat,Mmat,Mmat,Mmat,Mmat)
    del Mmat  
    # Convert Jacobian to dense, then to global matrix form, 
    # then solve the generalized eigenvalue problem
    dense_Jac = Jac.csr_rep.todense()
    print(f'Shape of local jacobian is: {dense_Jac.shape}', flush=True)
    print(f'Shape of global jacobian is: ({Jac.nrg}, {Jac.ncg})', flush=True)
    jac_gl = make_global_mat(dense_Jac, Jac.nrg, Jac.ncg, Jac.lg) # Generate global jacobian matrix
    mmat_gl = make_global_mat(Mmat_big, Jac.nrg, Jac.ncg, Jac.lg) # Generate global mass matrix
    del Jac, dense_Jac, Mmat_big
    print('Computing real jacobian matrix', flush=True)
    result = np.linalg.lstsq(jac_gl, mmat_gl) # invert global mass matrix and multiply by global jacobian to get real jacobian
    real_Jac = result[0]
    np.save('real_jacobian.npy', real_Jac)
  # plot the real jacobian matrix, eliminate values smaller than 1e-10 to improve contrast
  # real_Jac[np.abs(real_Jac)<1e-10] = 0
  # make the plot of the real jacobian matrix, use a logarithmic color scale to improve contrast
  # plt.imshow(real_Jac.real, aspect='auto', cmap='viridis', norm=plt.LogNorm(vmin=1e-10, vmax=np.max(np.abs(real_Jac.real))))
  # plt.colorbar()
  # plt.title('Real part of Jacobian')
  # plt.savefig('full/real_jacobian.png')
  # plt.clf()
  print('Computing eigenvalues', flush=True)
  w, v = sparse.linalg.eigs(real_Jac.real, k=6, sigma=1.01, which='LM')
  np.save('full/partial_eigs.npy', w)
  print(w, flush=True)
  plt.scatter(w.real, w.imag)
  plt.savefig('full/partial_spectrum.png')
  plt.clf()
  # R, C, sigmin = compute_pseudospectrum(real_Jac)
  # fig, ax = plt.subplots()
  # ax.scatter(w.real, w.imag)
  # artist = ax.contour(R, C, sigmin)
  # ax.clabel(artist, fontsize=10)
  # ax.set_title('Pseudospectrum')
  # plt.savefig('pseudoplot', format='png')