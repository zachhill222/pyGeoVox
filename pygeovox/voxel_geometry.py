import numpy as np

class VoxelGeometry:
	def __init__(self, N=[128, 128, 128], LOW=None, HIGH=None, dtype=np.double):
		self.dtype	= dtype
		if LOW is None: LOW = [0,0,0]
		if HIGH is None: HIGH = [1,1,1]
		self.LOW	= np.array(LOW, dtype=self.dtype)	#vertex 1 for the bounding box
		self.HIGH	= np.array(HIGH, dtype=self.dtype)	#vertex 2 for the bounding box
		self.markers = np.empty(N, dtype='i')

		#Coarsening averages adjacent cells and rounds down. it is probably best for the markers to be consecutive integers

	def setBounds(self, LOW=None, HIGH=None): #set specific box bounds and adjust them to match to the mesh
		if LOW is None:
			LOW = self.LOW
		else:
			self.LOW = np.array(LOW, dtype=self.dtype)
		
		if HIGH is None:
			HIGH = self.HIGH
		else:
			self.HIGH = np.array(HIGH, dtype=self.dtype)
		
	def trimFace(self, count, axis=None, side='BOTH'):
		if axis is None:
			axis = [0,1,2]
		elif not hasattr(axis, '__len__'):
			axis = [axis]

		for ii in axis:
			H = self.voxelSize()

			slc = [np.s_[:]]*3
			if side=='BOTH':
				slc[ii]=np.s_[count:self.markers.shape[ii]-count]
				self.LOW[ii] += (count*H[ii])
				self.HIGH[ii] -= (count*H[ii])
			elif side=='LOW':
				slc[ii]=np.s_[count::]
				self.LOW[ii] += (count*H[ii])
			elif side=='HIGH':
				slc[ii]=np.s_[:self.markers.shape[ii]-count]
				self.HIGH[ii] -= (count*H[ii])
			else:
				print('Unknown side to trim.')

			self.markers = self.markers[tuple(slc)]

	def extendFace(self, count, marker=0, axis=None, side='BOTH', method='pad'):
		if axis is None:
			axis = [0,1,2]
		elif not hasattr(axis, '__len__'):
			axis = [axis]

		if method=='extrude' and len(axis)>1:
			print("Extruding along multiple axes may lead to undefined behavior.")


		for ii in axis:
			H = self.voxelSize()

			slc = [np.s_[:]]*3
			if side=='BOTH' or side=='LOW':
				if method == 'extrude':
					slc = [np.s_[:]]*3
					slc[ii] = 0

					shape = [self.markers.shape[jj] for jj in range(3)]
					shape[ii] = 1

					block = self.markers[tuple(slc)].reshape(tuple(shape))
					block = np.repeat(block, count, axis=ii)
				else:
					padDim = [self.markers.shape[jj] for jj in range(3)]
					padDim[ii] = count

					block = np.empty(padDim, dtype='i')
					block.fill(marker)


				self.markers = np.concatenate((block, self.markers), axis=ii)
				self.LOW[ii] -= (count*H[ii])

			if side=='BOTH' or side=='HIGH':
				if method == 'extrude':
					slc = [np.s_[:]]*3
					slc[ii] = -1

					shape = [self.markers.shape[jj] for jj in range(3)]
					shape[ii] = 1

					block = self.markers[tuple(slc)].reshape(tuple(shape))
					block = np.repeat(block, count, axis=ii)
				else:
					padDim = [self.markers.shape[jj] for jj in range(3)]
					padDim[ii] = count

					block = np.empty(padDim, dtype='i')
					block.fill(marker)
				self.markers = np.concatenate((self.markers, block), axis=ii)
				self.HIGH[ii] += (count*H[ii])
			else:
				print('Unknown side to pad.')

			




	def refine(self, nLevels=1):
		for n in range(nLevels):
			# split each voxel into sub-voxels with edge-lengths one half of the original
			N = [2*n for n in self.markers.shape]
			newMarkers = np.empty(N, dtype='i')
			
			newMarkers[ ::2,  ::2,  ::2] = self.markers

			newMarkers[1::2,  ::2,  ::2] = self.markers
			newMarkers[ ::2, 1::2,  ::2] = self.markers
			newMarkers[ ::2,  ::2, 1::2] = self.markers

			newMarkers[ ::2, 1::2, 1::2] = self.markers
			newMarkers[1::2,  ::2, 1::2] = self.markers
			newMarkers[1::2, 1::2,  ::2] = self.markers

			newMarkers[1::2, 1::2, 1::2] = self.markers

			self.markers = newMarkers

	def coarsen(self, nLevels=1, defaultMarker=None):
		for n in range(nLevels):
			for ii in range(3):
				if self.markers.shape[ii]%2:
					print("The number of voxels in each dimension must be even in order to refine.")
					return
			

			N = [n//2 for n in self.markers.shape]
			meanMarker = np.zeros(N, dtype=np.double)
			
			meanMarker+= self.markers[ ::2,  ::2,  ::2]

			meanMarker+= self.markers[1::2,  ::2,  ::2]
			meanMarker+= self.markers[ ::2, 1::2,  ::2]
			meanMarker+= self.markers[ ::2,  ::2, 1::2]

			meanMarker+= self.markers[ ::2, 1::2, 1::2]
			meanMarker+= self.markers[1::2,  ::2, 1::2]
			meanMarker+= self.markers[1::2, 1::2,  ::2]

			meanMarker+= self.markers[1::2, 1::2, 1::2]

			if defaultMarker is None:
				meanMarker = meanMarker//8

				currentMarkers = np.unique(meanMarker.flatten())
				oldMarkers     = np.unique(self.markers.flatten()) #rounding may have introduced spurious markers
				for mkr in currentMarkers:
					if not mkr in oldMarkers:
						for ii in range(len(oldMarkers)-1):
							if oldMarkers[ii] <= mkr and mkr < oldMarkers[ii+1]:
								MKR = oldMarkers[ii]
								meanMarker[meanMarker==mkr] = MKR
			else:
				meanMarker = meanMarker/8
				currentMarkers = np.unique(meanMarker.flatten())
				oldMarkers     = np.unique(self.markers.flatten())
				for mkr in currentMarkers:
					if not mkr in oldMarkers:
						meanMarker[meanMarker==mkr] = defaultMarker

			self.markers = np.array(meanMarker, dtype='i')




	def __str__(self):
		string = []
		string.append("VoxelGeometry:\n")
		string.append("\tSize=\t[{:d}, {:d}, {:d}]\n".format(self.markers.shape[0], self.markers.shape[1], self.markers.shape[2]))
		string.append("\tXLIM=\t[{:+.6E},\t{:+.6E}]\t(delta= {:+.6E})\n".format(self.LOW[0], self.HIGH[0], self.HIGH[0]-self.LOW[0]))
		string.append("\tYLIM=\t[{:+.6E},\t{:+.6E}]\t(delta= {:+.6E})\n".format(self.LOW[1], self.HIGH[1], self.HIGH[1]-self.LOW[1]))
		string.append("\tZLIM=\t[{:+.6E},\t{:+.6E}]\t(delta= {:+.6E})\n".format(self.LOW[2], self.HIGH[2], self.HIGH[2]-self.LOW[2]))
		H = self.voxelSize()
		string.append("\tH=\t\t[{:+.6E},\t{:+.6E},\t{:+.6E}]\n".format(H[0], H[1], H[2]))

		uniqueMarkers = np.unique(self.markers.flatten())
		string.append("\t{} unique markers:".format(len(uniqueMarkers)))
		for mkr in uniqueMarkers:
			string.append("\n\t\t({}) count= {:7d}\tfraction= {:4E}".format(mkr, np.count_nonzero(self.markers==mkr), self.fraction(mkr)))

		return "".join(string)

	def nVoxels(self):
		return self.markers.size

	def resetMarkers(self, marker=0):
		self.markers = marker*np.ones(self.markers.shape, dtype=int)

	def fraction(self, marker=0):
		return np.count_nonzero(self.markers==marker)/self.markers.size

	def voxelSize(self):
		L = self.HIGH - self.LOW
		N = self.markers.shape
		if min(L) < 0:
			L *= 0.0
		return [L[ii]/N[ii] for ii in range(3)]

	def write2file(self, filename='Geometry.dat', writeheader=True):
		with open(filename,'w') as file:
				if writeheader:
					file.write('nx= {0:d}\n'.format(self.markers.shape[0]))
					file.write('ny= {0:d}\n'.format(self.markers.shape[1]))
					file.write('nz= {0:d}'.format(self.markers.shape[2]))

				for k in range(self.markers.shape[2]):
					block = '\n'
					for j in range(self.markers.shape[1]):
						for i in range(self.markers.shape[0]):
							block += '{0:d} '.format(self.markers[i,j,k])
						block += '\n'
					file.write(block)

	