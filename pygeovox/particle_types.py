import numpy as np
from scipy.special import beta as npBetaFun
from numba import njit

##### HELPER FUNCTIONS #####
#standard cross product in R3
@njit(cache=True)
def crossProd(x,y):
	return [x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0]]

#class to handle Quaternions
class Quaternion:
	def __init__(self, q0, qv, dtype=np.double):
		self.dtype = dtype

		#scalar part of quaternion
		self.q0 = self.dtype(q0)

		#vector part of quaternion
		self.qv = np.array(qv, dtype=self.dtype)

	def __str__(self):
		string = "Quaternion:\t({}, {}, {}, {})\n".format(self.q0, self.qv[0], self.qv[1], self.qv[2])
		return string

	def checkRotation(self):
		if np.abs(self.q0) > 1.0:
			return "NOT A VALID ROTATION"
		# get rotation theta/2
		half_theta = np.arccos(self.q0)
		if half_theta == 0:
			u = np.zeros(3, dtype='i')
		else:
			u = self.qv/np.sin(half_theta)
		norm_u = np.sum(u*u)

		return "Theta= {}\tU= [{}, {}, {}]\t|U|= {}".format(half_theta*2, u[0], u[1], u[2], np.sqrt(norm_u))

	#Define standard arithmetic operations for quaternions: these return a new copy of a quaternion
	def __mul__(self, other):
		#return self*other quaternion product
		q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		qv = self.q0*other.qv + other.q0*self.qv + crossProd(self.qv, other.qv)
		return Quaternion(q0, qv, self.dtype)

	def __rmul__(self, other):
		#return other*self quaternion product
		q0 = self.q0*other.q0 - self.qv.dot(other.qv)
		qv = self.q0*other.qv + other.q0*self.qv - crossProd(self.qv, other.qv)
		return Quaternion(q0, qv, self.dtype)

	def __add__(self, other):
		return Quaternion(self.q0+other.q0, self.qv+other.qv, dtype=self.dtype)

	def __sub__(self, other):
		return Quaternion(self.q0-other.q0, self.qv-other.qv, dtype=self.dtype)


	#Define important operations for quaternions
	def rotate(self, point, direction=1):
		v = Quaternion(0, point, dtype=self.dtype)
		if direction==1:
			return (self*v*self.conj()).qv
		elif direction==-1:
			return (self.conj()*v*self).qv

	def abs(self):
		qStarq = self.conj()*self
		return np.sqrt(qStarq.q0)

	def conj(self):
		return Quaternion(self.q0, -self.qv, self.dtype)

	def inv(self):
		qStar = self.conj()
		abs2  = (qStar*self).q0
		return Quaternion(qStar.q0/abs2, qStar.qv/abs2, dtype=self.dtype)

##### Shape3D CLASSES ####
# each class requires a self.contains(point) method and a self.LOW, self.HIGH bounding box
class Shape3D:
	def __init__(self, center=[0,0,0], bBoxOnly=False, dtype=np.double): #lower precision for faster arithmetic
		self.dtype 	= dtype
		self.LOW 	= np.array([9999999, 9999999, 9999999], dtype=dtype)
		self.HIGH 	= np.array([-9999999, -9999999, -9999999], dtype=dtype)
		self.center = np.array(center, dtype=dtype)

		# level set values for testing inclusion of points
		self.level_low  = -1
		self.level_high =  1

		# check if this particle has already been processed
		self.processed  = False

		# flag for checking bounding boxes
		self.bBoxOnly   = bBoxOnly



	def getVertexCount(self, globalLOW, H, globalShape): #coarseMask: known vertices contained in particel, coarseCount: lower bound on number of vertices in each voxel
		indLOW, indHIGH = self.getGlobalIndices(globalLOW=globalLOW, H=H, globalShape=globalShape)
		N = [indHIGH[ii]-indLOW[ii] for ii in range(3)]

		nVert = np.zeros(N, dtype='i')
		checkVert = np.ones([N[ii]+1 for ii in range(3)])

		for vertexInd in np.argwhere(checkVert):
			point = [globalLOW[ii]+H[ii]*(indLOW[ii]+vertexInd[ii]) for ii in range(3)]

			low   = [max(0,vertexInd[ii]-1) for ii in range(3)]
			high  = [min(N[ii], vertexInd[ii]) for ii in range(3)]
			slc   = [np.s_[low[ii]:high[ii]+1] for ii in range(3)]
			if self.contains(point):
				nVert[slc[0], slc[1], slc[2]] += 1

		self.processed = True

		return nVert


	def getGlobalIndices(self, globalLOW, H, globalShape): #voxel/LOW vertex indices
		indLOW  = [0,0,0]
		indHIGH = [0,0,0]
		for ii in range(3):
			low  = (self.LOW[ii]-globalLOW[ii])/H[ii]  
			high = (self.HIGH[ii]-globalLOW[ii])/H[ii] 
			indLOW[ii]  = max(int(np.floor(low)),0)
			indHIGH[ii] = min(int(np.ceil(high)), globalShape[ii])

			if indHIGH[ii] < indLOW[ii]:
				print(self)
				print(indLOW, indHIGH)

		return indLOW, indHIGH

	def coarseCheck(self, point): #make a coarse check and return point in local normalized corrdinates if it passes. overwrite for specific shapes.
		point = np.array(point, dtype=self.dtype)

		for ii in range(3):
			if point[ii] < self.LOW[ii] or point[ii] > self.HIGH[ii]:
				return False, None

		return True, point - self.center
		
	def evalLevelFun(self, point): #point must be in local (shifted and rotated) coordinates. overwrite this function for specific shapes
		pass

	def contains(self, point):
		passed, localPoint = self.coarseCheck(point)
		if self.bBoxOnly:
			return passed
		else:
			if passed:
				level_val = self.evalLevelFun(localPoint)
				return ( level_val >= self.level_low and level_val <= self.level_high )
			else:
				return False

	def paramString(self):
		string = []
		string.append("\tLOW=\t[{:+.6E}, {:+.6E}, {:+.6E}]\n".format(self.LOW[0], self.LOW[1], self.LOW[2]))
		string.append("\tHIGH=\t[{:+.6E}, {:+.6E}, {:+.6E}]\n".format(self.HIGH[0], self.HIGH[1], self.HIGH[2]))
		string.append("\tCENTER=\t[{:+.6E}, {:+.6E}, {:+.6E}]\n".format(self.center[0], self.center[1], self.center[2]))
		return "".join(string)

	def __str__(self):
		string = []
		string.append("Shape3D:\n")
		string.append(self.paramString)
		return "".join(string)



class Sphere(Shape3D):
	def __init__(self, radii=1, eps=None, center=[0,0,0],  quaternion=None, dtype=np.double):
		super().__init__(center=center, dtype=dtype)

		if hasattr(radii, '__len__'):
			self.radius = np.sum(radii)/len(radii)
		else:
			self.radius = radii

		self.LOW = self.center-1.05*self.radius
		self.HIGH = self.center+1.05*self.radius

	def coarseCheck(self, point): #performs a fast check and returns point in rotated coordinates if it passes
		# check if we are in the bounding box
		passed, point = super().coarseCheck(point)
		if not passed:
			return False, None

		# normalize
		point /= self.radius
		return True, point

	def evalLevelFun(self, point): #point in local normalized coordinates
		return sum(point*point)

	def exactVol(self):
		vol = (4*np.pi/3)
		for ii in range(3):
			vol*= self.radius
		return vol

	def __str__(self, label=""):
		string = []
		string.append("Sphere: {}\n".format(label))
		string.append(self.paramString())
		return "".join(string)

	def paramString(self):
		string = []
		string.append(super().paramString())
		string.append("\tRADIUS=\t{:+.6E}\n".format(self.radius))
		return "".join(string)



class Prism(Shape3D):
	def __init__(self, radii=[1,1,1], eps=None, center=[0,0,0], quaternion=[1,0,0,0], dtype=np.double):
		super().__init__(center=center, dtype=dtype)
		self.radii	= np.array(radii, dtype=self.dtype)		 #radii (rx, ry, rz)
		# self.QUAT   = np.array(quaternion, dtype=self.dtype) #quaternion for record keeping

		#rotation quaternion (q0, q1, q2, q3) where q0=cos(theta/2) and (q1,q2,q3)=sin(theta/2)*u
		#theta is the angle of rotation and u=(u1,u2,u3) is the axis of rotation and a unit vector
		#rotate a point P from the reference frame of the particle to standard reference frame by QPQ*
		self.quaternion	= Quaternion(quaternion[0], [quaternion[1], quaternion[2], quaternion[3]], dtype=self.dtype)

		point = np.array([0,0,0], dtype=self.dtype)
		padPercent = 0.05
		for x in range(2):
			point[0] = (-1)**(x%2) * (1+padPercent)*self.radii[0]
			for y in range(2):
				point[1] = (-1)**(y%2) * (1+padPercent)*self.radii[1]
				for z in range(3):
					point[2] = (-1)**(z%2) * (1+padPercent)*self.radii[2]

					rotPoint = self.center + self.quaternion.rotate(point)
					self.LOW = [min(self.LOW[ii], rotPoint[ii]) for ii in range(3)]
					self.HIGH = [max(self.HIGH[ii], rotPoint[ii]) for ii in range(3)]



	def evalLevelFun(self, point): #point in LOCAL NORMALIZED coordinates
		return max(np.abs(point))

	def exactVol(self):
		vol = 1.0
		for ii in range(3):
			vol*= (2*self.radii[ii])
		return vol

	def coarseCheck(self, point): #performs a fast check and returns point in rotated coordinates if it passes
		# check if we are in the bounding box
		passed, point = super().coarseCheck(point)
		if not passed:
			return False, None

		# rotate into local coordinates
		point = self.quaternion.rotate(point, direction=-1)

		# normalize
		point /= self.radii
		return True, point


	def __str__(self, label=""):
		string = []
		string.append("Prism: {}\n".format(label))
		string.append(self.paramString())
		return "".join(string)

	def paramString(self):
		string = []
		string.append(super().paramString())
		# string.append("\tQUAT=\t[{:.6E},\t{:.6E},\t{:.6E},\t{:.6E}]\n".format(self.QUAT[0], self.QUAT[1], self.QUAT[2], self.QUAT[3]))
		string.append("\tRADII=\t[{:+.6E}, {:+.6E}, {:+.6E}]\n".format(self.radii[0], self.radii[1], self.radii[2]))
		string.append("\t"+self.quaternion.__str__())
		string.append("\tRotation: {}\n".format(self.quaternion.checkRotation()))
		return "".join(string)



class Ellipsoid(Prism):
	def __init__(self, radii=[1,1,1], eps=None, center=[0,0,0], quaternion=[1,0,0,0], dtype=np.double):
		super().__init__(radii=radii, center=center, quaternion=quaternion, dtype=dtype)
		
	def evalLevelFun(self, point):#point in local reference frame and normalized by radii
		val = np.array(point, dtype=self.dtype) #point to evaluate
		val*=val								#square
		return sum(val)

	def exactVol(self):
		vol = 4*np.pi/3
		for ii in range(3):
			vol*= self.radii[ii]
		return vol

	def __str__(self, label=""):
		string = []
		string.append("Ellipsoid: {}\n".format(label))
		string.append(self.paramString())
		return "".join(string)

	def paramString(self):
		string = []
		string.append(super().paramString())
		return "".join(string)

class SuperEllipsoid(Ellipsoid):
	def __init__(self, radii=[1,1,1], eps=[1,1], center=[0,0,0], quaternion=[1,0,0,0],  dtype=np.double):
		super().__init__(radii=radii, center=center, quaternion=quaternion, dtype=dtype)
		self.eps	= np.array(eps, dtype=self.dtype)		 #shape parameters (eps1, eps2)

	def evalLevelFun(self, point): #point in local reference frame and normalized by radii
		val = point*point					#(x/r)^2
		val[0] = val[0]**(1.0/self.eps[1])	#(x/rx)^(2/eps2)
		val[1] = val[1]**(1.0/self.eps[1])	#(y/ry)^(2/eps2)
		val[2] = val[2]**(1.0/self.eps[0])	#(z/rz)^(2/eps1)
		return (val[0]+val[1])**(self.eps[1]/self.eps[0]) + val[2]

	def exactVol(self):
		vol = 2
		for ii in range(3):
			vol*= self.radii[ii]
		for ii in range(2):
			vol *= self.eps[ii]
		vol*= npBetaFun(0.5*self.eps[0], self.eps[0]+1)
		vol*=npBetaFun(0.5*self.eps[1], 0.5*self.eps[1]+1)
		return vol

	def __str__(self, label=""):
		string = []
		string.append("SuperEllipsoid: {}\n".format(label))
		string.append(self.paramString())
		return "".join(string)

	def paramString(self):
		string = []
		string.append(super().paramString())
		string.append("\tEPS=\t[{:+.6E}, {:+.6E}]\n".format(self.eps[0], self.eps[1]))
		return "".join(string)