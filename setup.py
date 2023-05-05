from setuptools import setup, find_packages

name = '191-final-project'
version = '0.1'
description = (
    "Implementation of HHL Algorithm"
)

with open("requirements.txt") as f:
    install_requires = f.read()
    
setup(
    name=name,
    version=version,
    description=description,
    install_requires=install_requires,
    packages=find_packages()
)