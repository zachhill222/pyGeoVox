from context import pyfem as pf
import numpy as np
from numba import njit
import scipy.sparse as sparse
import matplotlib.pyplot as plt

@njit
def defaultfun(x): return 0.0

class Heat(pf.Mesh1D.Galerkin1): #heat equation u_t - \laplace(u) = f(x,t)
	def __init__(self, nodes, fun=defaultfun):
		# initialize mesh
		super().__init__(nodes)
		
		# set problem parameters
		self.fun = fun

	def buildMats(self): #build standard mass and stiffness matrices (coo format and homogeneous Neumann BC)
		self.M = self.makeMassMat()
		self.A = self.makeStifMat()

	def setRandomIC(self, mean=0.0, std=0.3):
		self.u = mean + std*np.random.default_rng().standard_normal(self.nNode)
	
	def setIC(self, u):
		self.u = u
		
	def stepImplicit(self, dt, dirBC=[]):
		lhs = (self.M+dt*self.A).tolil()
		rhs = self.M.dot(dt*self.fun(self.node)+self.u)
		
		# set BC
		if len(dirBC) > 0:
			lhs[self.boundaryNode,:] = 0
			lhs[self.boundaryNode, self.boundaryNode] = 1.0
			rhs[self.boundaryNode] = dirBC

		self.u = sparse.linalg.spsolve(lhs.tocsr(),rhs)



if __name__ == '__main__':
	heat = Heat(np.linspace(0,1,100))
	heat.buildMats()
	heat.setRandomIC()

	for ii in range(1000):
		if not ii%100:
			plt.plot(heat.node, heat.u)
		heat.stepImplicit(0.0001, [1, heat.M.dot(heat.u).sum()/heat.measure()])

	plt.show()
	