from context import pygeovox as gv
from time import time

filename = './particles/particles_1000.txt'

mesh = gv.HexMesh3D(N=[1,1,1])
# mesh.bBoxOnly = True


print("Processing Geometry: ", end="")
t = time()
mesh.addParticlesFromFile(filename, pType='SuperEllipsoid')
mesh.processList()
print("{} seconds\n".format(time()-t))

print("Refining Geometry: ", end="")
t = time()
mesh.refine(5) #refining is faster than initializing a fine grid
print("{} seconds\n".format(time()-t))

print(mesh)
mesh.trimFace(15, axis=2)
mesh.cleanList()
# mesh.extendFace(10)
print(mesh)

mesh.refine(2)
mesh.extendFace(5, axis=2, method='extrude')


# print("Coarsening Geometry: ", end="")
# t = time()
# mesh.coarsen(1, process=True)
# print("{} seconds\n".format(time()-t))


mesh.meshRegion(mesh.ALLMARKER)
mesh.mesh2vtk('./outFiles/testAssemblyInterior.vtk')

mesh.meshRegion(mesh.SOMEMARKER)
mesh.mesh2vtk('./outFiles/testAssemblyBoundary.vtk')

print(mesh)