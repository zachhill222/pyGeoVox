from context import pyfem as pf
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from numba import njit


class Poisson(pf.Mesh1D.Galerkin1): #solve -\laplace(u)=f(x)
	def __init__(self, node, fval, weight=None):
		super().__init__(node)
		self.fval = fval
		self.w = weight

	def setDirichletBC(self, dirBC):
		self.lhs = self.makeStifMat(self.w).tolil()
		self.rhs = self.makeMassMat().dot(self.fval)

		self.lhs[self.boundaryNode,:] = 0
		self.lhs[self.boundaryNode,self.boundaryNode] = 1

		self.rhs[self.boundaryNode] = dirBC

	def solve(self):
		self.u = sparse.linalg.spsolve(self.lhs.tocsr(), self.rhs)

	def plot(self):
		plt.plot(self.node, self.u)
		plt.grid('on')
		plt.show()





if __name__ == '__main__':
	#define domain
	x = np.linspace(0,1,1000)
	x=x**3
	f = np.ones(len(x))

	@njit
	def w(x): return np.sin(np.pi*x)

	P = Poisson(x,f,w)
	P.setDirichletBC([0,0.25])
	P.solve()
	P.plot()
