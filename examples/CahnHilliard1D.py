from ...context import pyfem as pf
import numpy as np
from numba import njit
import scipy.sparse as sparse
import matplotlib.pyplot as plt

@njit
def defaultfun(u): return ((u-1)*(u+1))*u

@njit
def defaultFUN(u): return 0.25*(u-1)*(u-1)*(u+1)*(u+1)

class CahnHilliard(pf.Mesh1D.Galerkin1): #Cahn-Hilliard equation with smooth double well function and unit mobility: u_t - \laplace(w) = 0, w = f(u)/eps^2 - \laplace(u)
	def __init__(self, nodes, eps=0.1, fun=defaultfun, FUN=defaultFUN):
		# initialize mesh
		super().__init__(nodes)
		
		# set problem parameters
		self.ONEOVEREPS2 = 1.0/(eps*eps)
		self.fun = fun
		self.FUN = FUN #antiderivative of fun to be used in computing energy


	def buildMats(self): #build standard mass and stiffness matrices (coo format and homogeneous Neumann BC)
		self.M = self.makeMassMat()
		self.A = self.makeStifMat()

	def setRandomIC(self, mean=0.0, std=0.1):
		self.u = mean + std*np.random.default_rng().standard_normal(self.nNode)
		self.w = self.fun(self.u)*self.ONEOVEREPS2 - sparse.linalg.spsolve(self.M.tocsr(), self.A.dot(self.u))

	def setIC(self, u):
		self.u = u
		self.w = self.fun(self.u)*self.ONEOVEREPS2 - sparse.linalg.spsolve(self.M.tocsr(), self.A.dot(self.u))

	def calcEnergy(self):
		return 0.5*(self.A.dot(self.u).dot(self.u)) + self.ONEOVEREPS2*(self.M.dot(self.FUN(self.u)).sum())

	def calcEnergyRate(self):
		return -self.A.dot(self.w).dot(self.w)

	def stepIMEX(self, dt):
		lhs = sparse.block_array([[self.M, dt*self.A], [-self.A, self.M]])
		rhs = np.hstack([self.M.dot(self.u), self.ONEOVEREPS2*(self.M.dot(self.fun(self.u)))])

		uw  = np.split(sparse.linalg.spsolve(lhs.tocsr(), rhs), 2)

		self.u = uw[0]
		self.w = uw[1]

class CahnHilliardConvexSplit(pf.Mesh1D.Galerkin1): #Cahn-Hilliard equation with smooth double well function F(u)=0.25*(u^2-1)^2 and unit mobility: u_t - \laplace(w) = 0, w = f(u)/eps^2 - \laplace(u)
	# this class splits f(u)=u^3-u into the derivative of the convex part of F and the concave part of F
	def __init__(self, nodes, eps=0.05):
		# initialize mesh
		super().__init__(nodes)
		
		# set problem parameters
		self.ONEOVEREPS2 = 1.0/(eps*eps)
		self.fun = defaultfun #not used in time-stepping
		self.FUN = defaultFUN #antiderivative of fun_convex + fun_concave to be used in computing energy


	def buildMats(self): #build standard mass and stiffness matrices (coo format and homogeneous Neumann BC)
		self.M = self.makeMassMat()
		self.A = self.makeStifMat()

	def setRandomIC(self, mean=0.0, std=0.1):
		self.u = mean + std*np.random.default_rng().standard_normal(self.nNode)
		self.w = self.fun(self.u)*self.ONEOVEREPS2 - sparse.linalg.spsolve(self.M.tocsr(), self.A.dot(self.u))

	def setIC(self, u):
		self.u = u
		self.w = self.fun(self.u)*self.ONEOVEREPS2 - sparse.linalg.spsolve(self.M.tocsr(), self.A.dot(self.u))

	def calcEnergy(self):
		return 0.5*(self.A.dot(self.u).dot(self.u)) + self.ONEOVEREPS2*(self.M.dot(self.FUN(self.u)).sum())

	def calcEnergyRate(self):
		return -self.A.dot(self.w).dot(self.w)

	def stepIMEX(self, dt):
		lhs = sparse.block_array([[self.M, dt*self.A], [self.ONEOVEREPS2*self.M-self.A, self.M]])

		f_conv = self.u*self.u*self.u
		rhs = np.hstack([self.M.dot(self.u), self.ONEOVEREPS2*(self.M.dot(f_conv))])

		uw  = np.split(sparse.linalg.spsolve(lhs.tocsr(), rhs), 2)

		self.u = uw[0]
		self.w = uw[1]

if __name__ == '__main__':
	CH = CahnHilliard(np.linspace(0,1,100))
	CH.buildMats()
	CH.setRandomIC(1,.1)

	t = 0.0
	energy = [CH.calcEnergy()]
	time   = [t]
	

	for ii in range(10000):
		dEdt = CH.calcEnergyRate()
		dt = 0.001/np.sqrt(1+dEdt*dEdt)
		dt = np.max([dt, 1E-10])

		if not ii%100:
			plt.plot(CH.node, CH.u)

		CH.stepIMEX(dt)
		t+=dt

		energy.append(CH.calcEnergy())
		time.append(t)

	plt.show()
	
	plt.loglog(time, energy, '.k')
	plt.axhline(y=energy[0], color='k', linestyle='--')
	plt.show()

	plt.semilogy(time, '.k')
	plt.show()