from context import pygeovox as gv
import numpy as np
from time import time


theta = np.pi/4
axis = [0,1,1]
axisLen = np.sqrt(np.sum([axis[ii]*axis[ii] for ii in range(3)]))

q0 = np.cos(0.5*theta)
qv = [np.sin(0.5*theta)*axis[ii]/axisLen for ii in range(3)]

quat = [q0, qv[0], qv[1], qv[2]]



print("Processing Geometry: ", end="")
t = time()
mesh = gv.HexMesh3D(N=[8,8,8], dtype=np.double)
mesh.bBoxOnly = False

mesh.append(gv.Sphere(center=[0,0,-2], radii=[.25,.25,1], eps=[1,1.5], quaternion=quat))
mesh.append(gv.Prism(center=[0,0,0], radii=[1,1,1], eps=[1,1.5], quaternion=quat))
mesh.append(gv.Ellipsoid(center=[0,5,5], radii=[1,2,1], eps=[1,1.5], quaternion=quat))
mesh.append(gv.SuperEllipsoid(center=[-2,4,0], radii=[2,1,1], eps=[1.2,0.8], quaternion=quat))

mesh.processList()
print("{} seconds\n".format(time()-t))

print("Refining Geometry: ", end="")
t = time()
mesh.refine(3, process=True)
print("{} seconds\n".format(time()-t))


# print("Coarsening Geometry: ", end="")
# t = time()
# mesh.coarsen(1, process=True)
# print("{} seconds\n".format(time()-t))



mesh.meshRegion(mesh.ALLMARKER)
mesh.mesh2vtk('./outFiles/testParticleInterior.vtk')

mesh.meshRegion(mesh.SOMEMARKER)
mesh.mesh2vtk('./outFiles/testParticleBoundary.vtk')

print(mesh)
print(mesh.particleString())