from setuptools import setup, find_packages

setup(
	name= "pygeovox",
	version= "0.1",
	description= "A collection of FEM tools domains built from voxels.",
	author= "Zachary Hilliard",
	author_email= "zhilliard@regent.edu",
	packages= ["pygeovox"],
	install_requires= find_packages(exclude=('examples', 'tests'))
	)