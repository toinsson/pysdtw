#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path
import io

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

# pip's single-source version method as described here:
# https://python-packaging-user-guide.readthedocs.io/single_source_version/
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# create long description from readme for pypi
here = path.abspath(path.dirname(__file__))
with io.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='pysdtw',
      version=find_version('pysdtw', '__init__.py'),
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
