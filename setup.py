#!/usr/bin/env python
from distutils.core import setup
from setuptools import find_packages

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setup(name='pysdtw',
      version='0.0.1',
      description='Soft-DTW for CUDA devices.',
      author='Antoine Loriette',
      author_email='antoine.loriette@gmail.com',
      url='https://github.com/toinsson/pysdtw',
      packages=find_packages(exclude=['tests']),
      python_requires='>=3.7',
      install_requires=requirements.split("\n"),
     )
