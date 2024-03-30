from numba import njit, prange
import numpy as np
import scipy.sparse as sparse
from .particle_assembly import Assembly

class HexMesh3D(Assembly):
	def __init__(self, N=[128, 128, 128], LOW=None, HIGH=None, dtype=np.double):
		super().__init__(N=N, LOW=LOW, HIGH=HIGH, dtype=dtype)
		self.isMeshed = False

	def setBox(self, LOW, HIGH):
		super().setBox(LOW, HIGH)
		self.deleteMesh()

	def nElements(self):
		return self.element2node.shape[0]

	def nNodes(self):
		return self.node.shape[0]

	def __str__(self):
		string = []
		string.append("HexMesh3D:\n")
		if self.isMeshed:
			string.append("\tnNodes= {0}\n\tnElements= {1}\n".format(self.nNodes(), self.nElements()))
			string.append("\tnExtBoundNodes= {}\n".format(len(self.extBoundary)))
			string.append("\tnIntBoundNodes= {}\n".format(len(self.intBoundary)))
			string.append("\tnode_XLIM= [{:.6E},\t{:.6E}]\n".format(np.min(self.node[:,0]), np.max(self.node[:,0])))
			string.append("\tnode_YLIM= [{:.6E},\t{:.6E}]\n".format(np.min(self.node[:,1]), np.max(self.node[:,1])))
			string.append("\tnode_ZLIM= [{:.6E},\t{:.6E}]\n".format(np.min(self.node[:,2]), np.max(self.node[:,2])))

			eVol = self.discreteVolume()
			mVol, sVol = self.matrixVolume()
			mErr = (mVol-eVol)/eVol
			sErr = (sVol-eVol)/eVol

			string.append("\tmeshedVolume= {:3E}\n".format(eVol))
			string.append("\tmassMatrixVolume= {:3E} ({:%} error)\n".format(mVol, mErr))
			string.append("\tstifMatrixVolume= {:3E} ({:%} error)\n".format(sVol, sErr))
		else:
			 string.append("\tEmpty Mesh\n")

		string.append(super().__str__())

		return "".join(string)

	def matrixVolume(self):
		if self.isMeshed:
			M, A = self.makeBilinearMats()

			v1 = np.ones(self.nNodes())
			v2 = np.zeros(self.nNodes())
			for ii in range(3):
				v2+= self.node[:,ii] * (1.0/np.sqrt(3))

			massVol = self.Cmass * M.dot(v1).dot(v1)
			stifVol = self.Cstif * A.dot(v2).dot(v2)

			return massVol, stifVol

	def discreteVolume(self):
		if self.isMeshed:
			vol = self.nElements()
			H = self.voxelSize()
			for ii in range(3):
				vol*= H[ii]
			return vol

	def meshRegion(self, marker=None):
		if marker is None:
			marker = self.ALLMARKER

		#get mesh
		self.node, self.element2node, self.node2element, self.extBoundary, self.intBoundary, self.elementMarkers = mask2mesh3d_bool(self.markers == marker)
		self.elementMarkers = marker*self.elementMarkers #previous function returns markers of value 1. This corrects back to the original marker value.

		#correct node locations
		h = self.voxelSize()
		for ii in range(3):
			self.node[:,ii] *= h[ii]
			self.node[:,ii] += self.LOW[ii]

		if len(self.node) > 0:
			self.isMeshed = True
		else:
			self.isMeshed = False
	
	def refine(self, nLevels=1, process=True, nodeData=None, fluxData=None, cellData=None):
		super().refine(nLevels=nLevels, process=process) 

	def coarsen(self, nLevels=1, process=True, nodeData=None, fluxData=None, cellData=None):
		super().coarsen(nLevels=nLevels, process=process) 

	def interpData(self, xq, nodeData=None, fluxData=None, cellData=None):
		if not hasattr(xq,'__len__'):
			xq = [xq]

		elements = x2element(xq, self.node, self.element2node)
		print(elements)

	def meshAll(self):
		#get mesh
		self.node, self.element2node, self.node2element, self.extBoundary, self.intBoundary, self.elementMarkers = mask2mesh3d_all(self.markers)

		#correct node locations
		h = self.voxelSize()
		for ii in range(3):
			self.node[:,ii] *= h[ii]
			self.node[:,ii] += self.LOW[ii]

		if len(self.node) > 0:
			self.isMeshed = True
		else:
			self.isMeshed = False

	def deleteMesh(self):
		if self.isMeshed:
			del self.node 			
			del self.element2node 	
			del self.node2element 	
			del self.extBoundary  	
			del self.intBoundary  	
			del self.elementMarkers
		self.isMeshed = False

	def mesh2vtk(self, filename='mesh.vtk'):
		celldata = ('marker', self.elementMarkers)
		self.write2vtk(celldata=celldata, filename=filename, description="Solid Region")

	def write2vtk(self, pointdata=None, celldata=None, filename="data.vtk", description=""):
		# pointdata is a list of tuples ('variable name', variabledata[:])
		# celldata is a list of tuples ('variable name', variabledata[:])
		# each variabledata[ii] can be either a scalar or a vector
		if not self.isMeshed:
			print("Can't write VTK. There is no mesh!")
			return

		with open(filename, 'w') as file:
			###############  header ###############
			file.write("# vtk DataFile Version 2.0\n")
			file.write(description+"\n")
			file.write("ASCII\n\n")

			############## topology ###############
			file.write("DATASET UNSTRUCTURED_GRID\n")
			file.write("POINTS {0:d} float\n".format(self.nNodes()))
			string = []
			for i in range(self.nNodes()):
				for j in range(3): #we are in 3D for this class
					string.append("{0} ".format(self.node[i,j]))
				# if self.ndim == 2: #we are in 3D for this class
				# 	string += "0 "
				# string += "\n"
			string.append("\n")
			file.write(''.join(string))

			file.write("CELLS {0:d} {1:d}\n".format(self.nElements(), self.element2node.size + self.nElements()))
			string = []
			for i in range(self.nElements()):
				string.append("{0:d} ".format(len(self.element2node[i])))
				for j in range(len(self.element2node[i])):
					string.append("{0:d} ".format(self.element2node[i,j]))
				string.append("\n")
			string.append("\n")
			file.write(''.join(string))

			file.write("CELL_TYPES {0:d}\n".format(self.nElements()))
			# if self.ndim == 2:
			# 	string = "9\n"*self.nElements()
			# elif self.ndim == 3:
			string = []
			string.append("12\n"*self.nElements())
			string.append("\n")
			file.write(''.join(string))

			################ Point/Node Data #################
			if not pointdata is None:
				varname = pointdata[0]
				vardata = pointdata[1]

				# sanity check
				if not len(vardata) == self.nNodes():
					print("ERROR: data must be a list of scalars/vectors with the same length as nNodes\n")


				# set dimension of data
				if hasattr(vardata, '__len__'):
					dim = len(vardata)
				else:
					dim = 1 


				if dim == 1:
					file.write("POINT_DATA {0:d}\n".format(self.nNodes()))
					file.write("SCALARS " + varname + " double\n")
					file.write("LOOKUP_TABLE default\n")
					string = []
					for ii in range(len(vardata)):
						string.append("{0}\n".format(vardata[ii]))
					file.write(''.join(string))
				elif dim > 1:
					file.write("POINT_DATA {0:d}\n".format(self.nNodes()))
					file.write("VECTORS " + varname + " double\n")
					string = []
					for ii in range(len(vardata)):
						for k in range(dim):
							string.append("{0}\t".format(vardata[ii][k]))
						string.append("\n")
					file.write(''.join(string))

			################ Cell/Element Data #################
			if not celldata is None:
				varname = celldata[0]
				vardata = celldata[1]
				
				# sanity check
				if not len(vardata) == self.nElements():
					print("ERROR: data must be a list of scalars/vectors with the same length as nElements\n")

				# set dimension of data
				if hasattr(vardata[0], '__len__'):
					dim = len(vardata[0])
				else:
					dim = 1

				
				if dim == 1:
					file.write("CELL_DATA {0:d}\n".format(self.nElements()))
					file.write("SCALARS " + varname + " double\n")
					file.write("LOOKUP_TABLE default\n")
					string = []
					for ii in range(len(vardata)):
						string.append("{0}\n".format(vardata[ii]))
					file.write(''.join(string))
				elif dim > 1:
					file.write("CELL_DATA {0:d}\n".format(self.nElements()))
					file.write("VECTORS " + varname + " double\n")
					string = []
					for ii in range(len(vardata)):
						for k in range(dim):
							string.append("{0}\t".format(vardata[ii][k]))
						string.append("\n")
					file.write(''.join(string))

	def makeBilinearMats(self, Weights=[]):
		if len(Weights)==0:
			coo_data, self.Cmass, self.Cstif = makeStandardBilinearMats3d(self.element2node, self.voxelSize())
			massData = np.array(coo_data[:,0])
			stifData = np.array(coo_data[:,1])
			ii       = np.array(coo_data[:,2], dtype=int)
			jj       = np.array(coo_data[:,3], dtype=int)

		else:
			coo_data, self.Cmass, self.Cstif = makeIsotropicBilinearMats3d(self.element2node, self.voxelSize(), Weights)
			massData = np.array(coo_data[:,0])
			stifData = np.array(coo_data[:,1])
			ii       = np.array(coo_data[:,2], dtype=int)
			jj       = np.array(coo_data[:,3], dtype=int)


		M = sparse.coo_array((massData, (ii, jj)))
		A = sparse.coo_array((stifData, (ii, jj)))
		return M, A

#### COMPILED FUNCTIONS FOR MESHING ####
@njit(cache=True)
def mask2mesh3d_bool(mask):
	#mask should be a Nx-by-Ny-by-Nz boolean array with True/1 marking the voxels in the domain.
	#The node positions are assumed to be an integer in the range 0 <= x <= Nx and so on.
	#Because the mask must represent a uniform grid (i.e., all voxels are the same size), the positions can be post-processed to get the correct node vector.
	maxElements    = np.count_nonzero(mask) #number of voxels to include in the mesh
	maxNodes       = 8*maxElements #8 nodes per element, overestimate (depends on connectivity)
	maxExterior    = maxNodes
	maxInterior    = maxNodes

	
	# initialize node locations
	tempNodes      = np.empty((maxNodes, 3)) # list of node locations
	nodeIndex      = -1*np.ones((mask.shape[0]+1, mask.shape[1]+1, mask.shape[2]+1), dtype='i') #array to keep track of which nodes have been included in the mesh
	nNodes         = 0

	# initialize nodes in each element
	tempElements   = np.empty((maxElements, 8), dtype='i')
	nElements      = 0

	# initialize element markers
	tempElementMarkers = np.ones(maxElements, dtype='i')

	# exterior boundary nodes (if a node is on the boundary of the bounding box)
	tempExterior   = np.empty(maxExterior, dtype='i')
	nExterior      = 0

	# interior boundary nodes (if a node is part of a voxel flagged 1 and a voxel flagged 0)
	# exterior boundary takes priority (if a cell is part of the exterior boundary, it cannot be part of the interior boundary)
	tempInterior   = np.empty(maxInterior, dtype='i')
	nInterior      = 0

	# elements with each node
	tempNode2Elements = np.empty((maxNodes, 9), dtype='i') # [NO. OF ELEMENTS: element 1 , 2, 3, 4, ..., 9]

	ccw_ind = [0, 1, 4, 5, 3, 2, 7, 6] #standard index order for paraview
	# firstElement = True #make sure the first element and nodes are added correctly
	for index in np.argwhere(mask): #loop over included elements of the mask
		# loop through all nodes in element
		# print("==========")
		for k in range(2):
			for j in range(2):
				for i in range(2):
					# print(i,j,k, end="")
					# print(" | ", end="")
					# print(i+2*j+4*k, ccw_ind[i+2*j+4*k])

					# get index to check if node has been used
					node = np.array([index[0]+i, index[1]+j, index[2]+k])
					globalNodeNumber = nodeIndex[node[0], node[1], node[2]]

					# globalNodeNumber = in_array(tempNodes, node, nNodes)
					if globalNodeNumber == -1: #new node
						tempNodes[nNodes] = node
						globalNodeNumber = nNodes

						nodeIndex[node[0], node[1], node[2]] = globalNodeNumber
						nNodes += 1

						# check if node is on the exterior boundary
						onExterior = False
						if node[0] == 0 or node[0] == mask.shape[0]:
							tempExterior[nExterior] = globalNodeNumber
							nExterior += 1
							onExterior = True
						elif node[1] == 0 or node[1] == mask.shape[1]:
							tempExterior[nExterior] = globalNodeNumber
							nExterior += 1
							onExterior = True
						elif node[2] == 0 or node[2] == mask.shape[2]:
							tempExterior[nExterior] = globalNodeNumber
							nExterior += 1
							onExterior = True
						
						# check if node is on the interior boundary
						onInterior = False
						if not onExterior:
							# each element is 1x1x1, so node[] is the index of the element to the bottom left of node
							for ii in range(node[0]-1,node[0]+1):
								if onInterior: break
								for jj in range(node[1]-1,node[1]+1):
									if onInterior: break
									for kk in range(node[2]-1,node[2]+1):
										if onInterior: break
										if not mask[ii,jj,kk]: #there is a neighboring cell with 0 as its marker
											onInterior = True
							if onInterior:
								tempInterior[nInterior] = globalNodeNumber
								nInterior += 1

						# initialize node in to node2element matrix
						tempNode2Elements[globalNodeNumber,0] = 1
						tempNode2Elements[globalNodeNumber,1] = nElements
					else: #not a new node
						#add element to node2element matrix
						tempNode2Elements[globalNodeNumber,0] += 1
						ind = tempNode2Elements[globalNodeNumber,0]
						tempNode2Elements[globalNodeNumber,ind] = nElements


					# add node to element
					tempElements[nElements,ccw_ind[i+2*j+4*k]] = globalNodeNumber
		nElements += 1
		firstElement = False
	
	return tempNodes[0:nNodes], tempElements[0:nElements], tempNode2Elements[0:nNodes], tempExterior[0:nExterior], tempInterior[0:nInterior], tempElementMarkers[0:nElements]

@njit(cache=True)
def mask2mesh3d_all(markers): #TODO speedup this algorithem. Use color markings to make parallel?
	#markers should be a Nx-by-Ny-by-Nz integer array with flags marking the voxels in the domain.
	#The node positions are assumed to be an integer in the range 0 <= x <= Nx and so on.
	#Because the mask markers array represents a uniform grid (i.e., all voxels are the same size), the positions can be post-processed to get the correct node vector.
	#The entire hexahedral domain will be meshed, with each element recieving a marker (i.e., its flag in the markers input array)
	maxElements    = markers.size #number of voxels to include in the mesh
	
	maxNodes       = 1
	for n in markers.shape: maxNodes*=(n+1)

	maxInterior    = 1
	for n in markers.shape: maxInterior*=(n-1)

	maxExterior    = maxNodes - maxInterior

	# print(maxNodes, maxExterior, maxInterior)
	
	# initialize node locations
	tempNodes      = np.empty((maxNodes, 3)) # list of node locations
	nodeIndex      = -1*np.ones((markers.shape[0]+1, markers.shape[1]+1, markers.shape[2]+1), dtype='i') #array to keep track of which nodes have been included in the mesh
	nNodes         = 0

	# initialize nodes in each element
	tempElements   = np.empty((maxElements, 8), dtype='i')
	nElements      = 0

	# initialize element markers
	tempElementMarkers = np.empty(maxElements, dtype='i')

	# exterior boundary nodes (if a node is on the boundary of the bounding box)
	tempExterior   = np.empty(maxExterior, dtype='i')
	nExterior      = 0

	# interior boundary nodes (none, only used for having consistent arguments)
	tempInterior   = np.empty(0, dtype='i')

	# elements with each node
	tempNode2Elements = np.empty((maxNodes, 9), dtype='i') # [NO. OF ELEMENTS: element 1 , 2, 3, 4, ..., 9]

	ccw_ind = [0, 1, 4, 5, 3, 2, 7, 6] #standard index order for paraview
	# firstElement = True #make sure the first element and nodes are added correctly
	for ii in range(markers.shape[0]): #loop over included elements of the mask
		for jj in range(markers.shape[1]): #loop over included elements of the mask
			for kk in range(markers.shape[2]): #loop over included elements of the mask
				index = [ii, jj, kk]

				# loop through all nodes in element
				# print("==========")
				for k in range(2):
					for j in range(2):
						for i in range(2):
							# print(i,j,k, end="")
							# print(" | ", end="")
							# print(i+2*j+4*k, ccw_ind[i+2*j+4*k])

							# get index to check if node has been used
							node = np.array([index[0]+i, index[1]+j, index[2]+k])
							globalNodeNumber = nodeIndex[node[0], node[1], node[2]]

							# globalNodeNumber = in_array(tempNodes, node, nNodes)
							if globalNodeNumber == -1: #new node
								tempNodes[nNodes] = node
								globalNodeNumber = nNodes

								nodeIndex[node[0], node[1], node[2]] = globalNodeNumber
								nNodes += 1

								# check if node is on the exterior boundary
								if node[0] == 0 or node[0] == markers.shape[0]:
									tempExterior[nExterior] = globalNodeNumber
									nExterior += 1
								elif node[1] == 0 or node[1] == markers.shape[1]:
									tempExterior[nExterior] = globalNodeNumber
									nExterior += 1
								elif node[2] == 0 or node[2] == markers.shape[2]:
									tempExterior[nExterior] = globalNodeNumber
									nExterior += 1
								
								

								# initialize node in to node2element matrix
								tempNode2Elements[globalNodeNumber,0] = 1
								tempNode2Elements[globalNodeNumber,1] = nElements
							else: #not a new node
								#add element to node2element matrix
								tempNode2Elements[globalNodeNumber,0] += 1
								ind = tempNode2Elements[globalNodeNumber,0]
								tempNode2Elements[globalNodeNumber,ind] = nElements


							# add node to element
							tempElements[nElements,ccw_ind[i+2*j+4*k]] = globalNodeNumber

				# save marker for element
				tempElementMarkers[nElements] = markers[ii,jj,kk]
				nElements += 1
	return tempNodes[0:nNodes], tempElements[0:nElements], tempNode2Elements[0:nNodes], tempExterior[0:nExterior], tempInterior, tempElementMarkers[0:nElements]


#### COMPILED FUNCTIONS FOR CREATING MATRICES ####
@njit(parallel=True, cache=True)
def makeStandardBilinearMats3d(elements, H):
	#uniform hexahedral voxels of size H[0] by H[1] by H[2]
	nElements = elements.shape[0]
	coo_data  = np.empty((64*nElements, 4))

	# reference element corners in standard (Paraview) orientation (https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html)
	eta_ccw   = [-1, -1, -1, -1,  1,  1,  1,  1] #X
	xi_ccw    = [-1,  1,  1, -1, -1,  1,  1, -1] #Y
	zeta_ccw  = [-1, -1,  1,  1, -1, -1,  1,  1] #Z
	ij2index  = np.arange(64).reshape(8,8)

	# constant to avoid divisions
	Cmass = (H[0]*H[1]*H[2]) / (12.0**3)
	Cstif = (H[0]*H[1]*H[2]) / (12.0**2)
	oneOverH = [1.0/(h*h) for h in H]

	for elem in prange(nElements):
		block_start = 64*elem
		for i in range(8):
			node1 = elements[elem,i]
			eta1  = eta_ccw[i]
			xi1   = xi_ccw[i]
			zeta1 = zeta_ccw[i]

			XX    = xi1*xi1
			YY    = eta1*eta1
			ZZ    = zeta1*zeta1

			# compute diagonal element
			massVal           = (3+XX)*(3+YY)*(3+ZZ) #*Cmass
			# massVal *= Cmass

			stifVal           = (oneOverH[0]*XX)*(3+YY)*(3+ZZ) + (3+XX)*(oneOverH[1]*YY)*(3+ZZ) + (3+XX)*(3+YY)*(oneOverH[2]*ZZ) #*Cstif
			# stifVal *= Cstif

			index             = block_start + ij2index[i,i]
			coo_data[index,0] = massVal
			coo_data[index,1] = stifVal
			coo_data[index,2] = node1
			coo_data[index,3] = node1

			# compute off-diagonal elements
			for j in range(i+1,8):
				node2 = elements[elem,j]
				eta2  = eta_ccw[j]
				xi2   = xi_ccw[j]
				zeta2 = zeta_ccw[j]

				XX    = xi1*xi2
				YY    = eta1*eta2
				ZZ    = zeta1*zeta2

				# compute diagonal element
				massVal           = (3+XX)*(3+YY)*(3+ZZ) #*Cmass
				# massVal *= Cmass

				stifVal           = (oneOverH[0]*XX)*(3+YY)*(3+ZZ) + (3+XX)*(oneOverH[1]*YY)*(3+ZZ) + (3+XX)*(3+YY)*(oneOverH[2]*ZZ) #*Cstif
				# stifVal *= Cstif

				index             = block_start + ij2index[i,j]
				coo_data[index,0] = massVal
				coo_data[index,1] = stifVal
				coo_data[index,2] = node1
				coo_data[index,3] = node2

				index             = block_start + ij2index[j,i]
				coo_data[index,0] = massVal
				coo_data[index,1] = stifVal
				coo_data[index,2] = node2
				coo_data[index,3] = node1

	return coo_data, Cmass, Cstif

@njit(parallel=True, cache=True)
def makeIsotropicBilinearMats3d(elements, H, Weights):
	#uniform hexahedral voxels of size H[0] by H[1] by H[2]
	#non-homogenious, but isotropic weighting function Weights[:,0] for mass and Weights[:,1] for stiffness
	nElements = elements.shape[0]
	coo_data  = np.empty((64*nElements, 4))

	# reference element corners in counter-clockwise orientation
	eta_ccw   = [-1, -1, -1, -1,  1,  1,  1,  1] #X
	xi_ccw    = [-1,  1,  1, -1, -1,  1,  1, -1] #Y
	zeta_ccw  = [-1, -1,  1,  1, -1, -1,  1,  1] #Z
	ij2index  = np.arange(64).reshape(8,8)

	# constant to avoid divisions
	Cmass = (H[0]*H[1]*H[2]) / (12**3)
	Cstif = (H[0]*H[1]*H[2]) / (12**2)
	oneOverH = [1.0/(h*h) for h in H]

	for elem in prange(nElements):
		block_start = 64*elem
		for i in range(8):
			node1 = elements[elem,i]
			eta1  = eta_ccw[i]
			xi1   = xi_ccw[i]
			zeta1 = zeta_ccw[i]

			XX    = xi1*xi1
			YY    = eta1*eta1
			ZZ    = zeta1*zeta1

			# compute diagonal element
			massVal           = (3+XX)*(3+YY)*(3+ZZ) #*Cmass
			massVal *= (Weights[elem,0])

			stifVal           = (oneOverH[0]*XX)*(3+YY)*(3+ZZ) + (3+XX)*(oneOverH[1]*YY)*(3+ZZ) + (3+XX)*(3+YY)*(oneOverH[2]*ZZ) #*Cstif
			stifVal *= (Weights[elem,1])

			index             = block_start + ij2index[i,i]
			coo_data[index,0] = massVal
			coo_data[index,1] = stifVal
			coo_data[index,2] = node1
			coo_data[index,3] = node1

			# compute off-diagonal elements
			for j in range(i+1,8):
				node2 = elements[elem,j]
				eta2  = eta_ccw[j]
				xi2   = xi_ccw[j]
				zeta2 = zeta_ccw[j]

				XX    = xi1*xi2
				YY    = eta1*eta2
				ZZ    = zeta1*zeta2

				# compute diagonal element
				massVal           = (3+XX)*(3+YY)*(3+ZZ) #*Cmass
				massVal *= (Weights[elem,0])

				stifVal           = (oneOverH[0]*XX)*(3+YY)*(3+ZZ) + (3+XX)*(oneOverH[1]*YY)*(3+ZZ) + (3+XX)*(3+YY)*(oneOverH[2]*ZZ) #*Cstif
				stifVal *= (Weights[elem,1])

				index             = block_start + ij2index[i,j]
				coo_data[index,0] = massVal
				coo_data[index,1] = stifVal
				coo_data[index,2] = node1
				coo_data[index,3] = node2

				index             = block_start + ij2index[j,i]
				coo_data[index,0] = massVal
				coo_data[index,1] = stifVal
				coo_data[index,2] = node2
				coo_data[index,3] = node1

	return coo_data, Cmass, Cstif


#### COMPILED FUNCTIONS FOR INTERPOLATING ####
@njit(parallel=True, cache=True)
def x2element(xList, node, element2node):
	#a point could be in 8 elements if it is a node. the value K=elList[n,0] is the number of elements that x[n] is in.
	# the values elList[n,1:1+K] are the elements that x[n] are in.
	elList = np.zeros((len(xList),9),dtype='i')

	#race conditions don't matter if x is on the boundary of two elements if the function to be interpolated is continuous
	for el in prange(len(element2node)):
		# get bounding box for element
		localNodes = element2node[el]
		low  = [node[localNodes[0], ii] for ii in range(3)]
		high = [node[localNodes[6], ii] for ii in range(3)]

		for n in prange(len(elList)):
			x = xList[n]
			inEl = True
			for ii in range(3):
				if x[ii] < low[ii] or x[ii] < high[ii]:
					inEl = False
					break
			if inEl:
				K = elList[n,0]
				elList[n,0] += 1
				elList[n,K+1] = el

	return elList

