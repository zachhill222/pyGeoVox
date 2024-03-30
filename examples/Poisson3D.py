from context import pygeovox as gv
from time import time
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt


particle_file = './particles/particles.txt'

print("Setting up geometry: ", end="")
t   = time()
mesh = gv.voxel3d.HexMesh3D(N=[32, 32, 32])
# mesh.setBox(LOW=[-1.25 for i in range(3)], HIGH =[1.25 for i in range(3)])
mesh.addSuperEllipsoidFromFile(particle_file)
# mesh.append(gv.voxel3d.Sphere(0.5, [0,0,0]))
mesh.processList(marker=1, level_low=0, level_high=1.0, method='all')
print("\t{} seconds".format(time()-t))
t = time()

print("Meshing geometry: ", end="")
# mesh.meshRegion(0)
mesh.meshAll()
print("\t{} seconds".format(time()-t))
print(mesh)
t = time()

print("Creating Mass and Stiffness Matrices: ", end="")
weights = np.ones((mesh.nElements(),2))
# weights[mesh.elementMarkers==1,1] = 100000
M, A = mesh.makeBilinearMats(Homogeneous=False, Weights=weights)
print("\t{} seconds".format(time()-t))
t = time()


print("Setting Up Laplace Equation: ", end="")
LHS = mesh.Cstif*A


sourceInd = np.unique(mesh.element2node[mesh.elementMarkers==1].flatten())

RHS = 0*np.ones(mesh.nNodes(), dtype=mesh.dtype)
RHS[sourceInd] = 10
RHS = mesh.Cmass*M.dot(RHS)


LHS = LHS.tolil(copy=False)
ind = mesh.extBoundary[mesh.node[mesh.extBoundary,0]==mesh.LOW[0]]
LHS[ind,:]   = 0
LHS[ind,ind] = 1
RHS[ind]     = 0

ind = mesh.extBoundary[mesh.node[mesh.extBoundary,1]==mesh.LOW[1]]
LHS[ind,:]   = 0
LHS[ind,ind] = 1
RHS[ind]     = 0

ind = mesh.extBoundary[mesh.node[mesh.extBoundary,2]==mesh.LOW[2]]
LHS[ind,:]   = 0
LHS[ind,ind] = 1
RHS[ind]     = 0


LHS = LHS.tocsr(copy=False)

print(" {} seconds".format(time()-t))
t = time()

print("Solving Laplace Equation: ", end="")
u = sparse.linalg.spsolve(LHS, RHS)
filename = './poisson3d_solution.vtk'
mesh.write2vtk(pointdata=('temperature', u), celldata=('marker', mesh.elementMarkers), filename=filename)
print("{} seconds".format(time()-t))
