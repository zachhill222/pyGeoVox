import numpy as np
from .voxel_geometry import *
from .particle_types import *


class Assembly(VoxelGeometry):
	def __init__(self, N=[128, 128, 128], LOW=None, HIGH=None, dtype=np.half):
		self.list = []
		if LOW is None: LOW   = [ 99999,  99999,  99999]
		if HIGH is None: HIGH = [-99999, -99999, -99999]
		super().__init__(N=N, LOW=LOW, HIGH=HIGH, dtype=dtype)

		# set marker types they must all be integers
		# with coarsening, it is important that NONE < SOME < ALL
		self.NONEMARKER = 0
		self.SOMEMARKER = self.NONEMARKER + 1
		self.ALLMARKER  = self.NONEMARKER + 2

		self.markers.fill(self.NONEMARKER)

		# toggle to use bounding box only. helpful for debugging.
		self.bBoxOnly = False

	def append(self, shape):
		#append any object with inheritance from Shape3D
		self.list.append(shape)
		
		# adjust bounding box
		for ii in range(3):
			self.LOW[ii]  = min(self.LOW[ii],  shape.LOW[ii])
			self.HIGH[ii] = max(self.HIGH[ii], shape.HIGH[ii])
	
	def exactVol(self):
		vol = 0.0
		for particle in self.list:
			vol+= particle.exactVol()
		return vol

	def discreteVol(self):
		lowVol  = self.fraction(marker=self.ALLMARKER)
		highVol = lowVol + self.fraction(marker=self.SOMEMARKER)

		for ii in range(3):
			lowVol *= (self.HIGH[ii]-self.LOW[ii])
			highVol*= (self.HIGH[ii]-self.LOW[ii])

		return lowVol, highVol

	def intersects(self, particle):
		# particle is a Shape3D with a LOW and HIGH bounding box
		intersects = True
		for ii in range(3):
			if particle.LOW[ii] > self.HIGH[ii] or particle.HIGH[ii] < self.LOW[ii]:
				intersects = False
		return intersects

	def cleanList(self):
		keepInd = []
		for p in range(len(self.list)):
			if self.intersects(self.list[p]):
				keepInd.append(p)

		newList = []
		for p in keepInd:
			newList.append(self.list[p])

		self.list = newList

	def resetProcessedFlags(self):
		for particle in self.list:
			particle.processed = False

	def extendFace(self, count, marker=None, reProcess=None, axis=None, side='BOTH', method='pad'):
		if reProcess is None:
			if method == 'pad':
				reProcess = True
			elif method == 'extrude':
				reProcess = False

		if marker is None:
			if reProcess:
				marker = self.SOMEMARKER
			else:
				marker = self.NONEMARKER

		if reProcess and method == 'extrude':
			print("Is re-processing after extruding what you want?")

		super().extendFace(count=count, marker=marker, axis=axis, side=side, method=method)
		if reProcess: self.processList()

	def addParticlesFromFile(self, filename, pType='Sphere'):
		with open(filename, 'r') as file:
			# skip over comments in header
			line = "#"
			while line[0] == "#":
				line = file.readline()

			# add shapes to list
			while line != "":
				par    = line.split('\t')
				
				radii  = [float(par[0]), float(par[1]), float(par[2])]
				eps    = [float(par[3]), float(par[4])]
				center = [float(par[5]), float(par[6]), float(par[7])]
				quat   = [float(par[8]), float(par[9]), float(par[10]), float(par[11])]
				
				if pType == 'Sphere':
					self.append(Sphere(radii, eps, center, quat))
				elif pType == 'Prism':
					self.append(Prism(radii, eps, center, quat))
				elif pType == 'Ellipsoid':
					self.append(Ellipsoid(radii, eps, center, quat))
				elif pType == 'SuperEllipsoid':
					self.append(SuperEllipsoid(radii, eps, center, quat))
				else:
					print("Particle type not supported.")
					break

				line   = file.readline()
				

	def processList(self, checkAll=False):
		for particle in self.list: particle.bBoxOnly = self.bBoxOnly

		H = self.voxelSize()
		tmpNONEMARKER = min([self.NONEMARKER, self.SOMEMARKER, self.ALLMARKER]) - 1
		voxelAssigned = np.zeros(self.markers.shape, dtype='bool')
		voxelAssigned[(self.markers==self.ALLMARKER)] = True

		for particle in self.list:
			if not self.intersects(particle): continue


			# print(particle)
			indLOW, indHIGH = particle.getGlobalIndices(globalLOW=self.LOW, H=H, globalShape=self.markers.shape)
			N = [indHIGH[ii]-indLOW[ii] for ii in range(3)]

			nVert = np.zeros(N, dtype='i')

			# pre-process vertex mask
			localMarkerView = self.markers[indLOW[0]:indHIGH[0], indLOW[1]:indHIGH[1], indLOW[2]:indHIGH[2]].view()
			localAssignedView = voxelAssigned[indLOW[0]:indHIGH[0], indLOW[1]:indHIGH[1], indLOW[2]:indHIGH[2]].view()
			localAssignedView[(localMarkerView==self.ALLMARKER)] = True

			# print(indLOW, indHIGH, N, nVert.shape, localMarkerView.shape)

			if particle.processed and not checkAll and np.count_nonzero(localMarkerView==self.ALLMARKER)>64 and np.count_nonzero(localMarkerView==self.SOMEMARKER)>64:
				checkVert = np.zeros([N[ii]+1 for ii in range(3)])

				#only need to check SOMEMARKER voxels (particles are convex)
				for voxelInd in np.argwhere((localMarkerView == self.SOMEMARKER) | (localMarkerView==tmpNONEMARKER)):
					low  = [voxelInd[ii] for ii in range(3)]
					high = [voxelInd[ii]+1 for ii in range(3)]
					slc  = [np.s_[low[ii]:high[ii]+1] for ii in range(3)]
					# print(voxelInd, slc)
					checkVert[slc[0], slc[1], slc[2]] = True

				# loop through all vertices
				nVert = np.zeros(N, dtype='i')
				for vertexInd in np.argwhere(checkVert):
					point = [self.LOW[ii]+H[ii]*(indLOW[ii]+vertexInd[ii]) for ii in range(3)]

					low   = [max(0,vertexInd[ii]-1) for ii in range(3)]
					high  = [min(N[ii], vertexInd[ii]) for ii in range(3)]
					slc   = [np.s_[low[ii]:high[ii]+1] for ii in range(3)]

					checkVert[vertexInd[0], vertexInd[1], vertexInd[2]] = False
					if particle.contains(point):
						nVert[slc[0], slc[1], slc[2]] += 1

				# update voxels
				indAssign = (nVert >= 8)
				localMarkerView[indAssign] = self.ALLMARKER
				localAssignedView[indAssign] = True

				indAssign = (nVert<8) & (nVert>0) & (localAssignedView==0)
				localMarkerView[indAssign] = self.SOMEMARKER
				localAssignedView[indAssign] = True

				indAssign = (nVert==0) & (localAssignedView==0)
				localMarkerView[indAssign] = tmpNONEMARKER

				# print(localMarkerView.shape)

				
			else:
				nVert = particle.getVertexCount(globalLOW=self.LOW, H=H, globalShape=self.markers.shape)
				
				indAssign = (nVert >= 8)
				localMarkerView[indAssign] = self.ALLMARKER
				localAssignedView[indAssign] = True

				indAssign = (nVert<8) & (nVert>0) & (localAssignedView==0)
				localMarkerView[indAssign] = self.SOMEMARKER
				localAssignedView[indAssign] = True

				indAssign = (nVert==0) & (localAssignedView==0)
				localMarkerView[indAssign] = tmpNONEMARKER

		# final clean up: assign NONMARKERS from tmpNONEMARKER
		# when refining, it is possible that an old SOMEMARKER is no longer in bounding boxes and thus not checked. set these to NONEMARKER
		indAssign = (self.markers==tmpNONEMARKER) | (self.markers==self.SOMEMARKER & ~voxelAssigned)
		self.markers[indAssign] = self.NONEMARKER


	def refine(self, nLevels=1, process=True):
		for n in range(nLevels):
			super().refine()
			if process: self.processList()

	
	def coarsen(self, nLevels=1, process=True):
		super().coarsen(nLevels=nLevels, defaultMarker=self.SOMEMARKER)
		if process: self.processList()

	def __str__(self, label="", listParticles=False):
		string = []
		string.append("Assembly: {}\n".format(label))
		string.append("\tnShapes= {}\n".format(len(self.list)))

		eVol = self.exactVol()
		lVol, hVol = self.discreteVol()
		lErr = (lVol-eVol)/eVol
		hErr = (hVol-eVol)/eVol
		meanVol = 0.5*(lVol+hVol)
		meanErr = 0.5*(lErr+hErr)
		string.append("\texactVolume= {:3E}\n".format(eVol))
		string.append("\tlowerDiscreteVolume= {:3E} ({:%} error)\n".format(lVol, lErr))
		string.append("\tupperDiscreteVolume= {:3E} ({:%} error)\n".format(hVol, hErr))
		string.append("\tmeanDiscreteVolume= {:3E} ({:%} error)\n".format(meanVol, meanErr))

		if listParticles:
			string.append(self.particleString())

		string.append(super().__str__())
		return "".join(string)


	def particleString(self):
		string = []
		for ii in range(len(self.list)):
			string.append(self.list[ii].__str__(label=ii)+"\n")
		return "".join(string)
		