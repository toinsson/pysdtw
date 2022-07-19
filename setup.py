#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
import io

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()


# create long description from readme for pypi
here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pysdtw',
      version='0.0.4',
      description='Torch implementation of Soft-DTW, supports CUDA devices.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Antoine Loriette',
      author_email='antoine.loriette@gmail.com',
      url='https://github.com/toinsson/pysdtw',
      packages=find_packages(exclude=['tests']),
      python_requires='>=3.7',
      install_requires=requirements.split("\n"),
     )
