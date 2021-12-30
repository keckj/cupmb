<!--
Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved

Project CUPMB and package cupmb are proprietary and confidential
and can not be copied, reproduced, modified and/or distributed without the express permission
of all copyright holders (Jean-Baptiste Keck).

Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
-->
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)]()
[![Python 3.8](https://img.shields.io/badge/python-3.8--3.9-blue.svg)](https://www.python.org/downloads/)
[![Cuda 11](https://img.shields.io/badge/cuda-11.0--11.4-green.svg)](https://developer.nvidia.com/cuda-toolkit-archive)
[![Licence](https://img.shields.io/badge/license-proprietary-orange)](https://en.wikipedia.org/wiki/Proprietary_software)
[![Pipeline Status](https://gitlab.com/keckj/smpost/badges/main/pipeline.svg)](https://gitlab.com/keckj/smpost/-/commits/main)
[![Coverage Report](https://gitlab.com/keckj/smpost/badges/main/coverage.svg)](https://gitlab.com/keckj/smpost/-/commits/main)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Cuda Parallel Marching Blocks

A Cuda implementation of the Parallel Marching Blocks algorithm as a Python3 module. If you are looking for an efficient marching squares / marching cubes implementation on the GPU, just give it a try. See [Parallel Marching Blocks: A Practical Isosurfacing Algorithm for Large Data on Many-Core Architectures](https://www.researchgate.net/publication/304811277_Parallel_Marching_Blocks_A_Practical_Isosurfacing_Algorithm_for_Large_Data_on_Many-Core_Architectures) for more information about the algorithm.

## Features:
- Compatible with both [Cupy](https://cupy.dev/) and [PyCuda](https://documen.tician.de/pycuda/) backends.
- 1D, 2D and 3D support, single or double precision for vertices.
- Supports signed/unsigned char/short/int/longlong for simplices.
- Scalar field can be analytic (possibly generated at runtime) or stored in an array/texture.
- Low memory consumption with user configurable allocation policies (and possibly user provided buffers).
- Handling of custom user defined vertex properties and user provided extra kernel parameters.
- Support for dynamic domains (axis-aligned bounding box) and dynamic kernel parameters.
- Compatible with Cuda/OpenGL interoperability ([currently only available for the PyCuda backend](https://giters.com/cupy/cupy/issues/5711))
- Vertices and simplices are stored as array of structures, vector loads and stores are optimized with  [trove](https://github.com/bryancatanzaro/trove).

## Examples
- `0_simple.py` This example will show you how to execute the marching cubes algorithm to find the zero levelset of a scalar field representing a sphere. The scalar field is defined as $\phi(x,y,z) = x^2 + y^2 + z^2 - r^2$ where $r=0.5$ is a constant defined directly in the device scalar function. The default bounding box is $[0,1]^3$ so we offset it by $x_0=-r$ to find the zeros for $(x,y,z) \in [-0.5,+0.5]^3$. The output mesh (vertices `V`, simplices `I`) is copied to the host (numpy arrays) and is then displayed by using the [trimesh](https://trimsh.org/trimesh.html) module. Parameter `N` controls the resolution of the marching cubes algorithm.
# How to get started


```
git clone --recursive https://gitlab.com/keckj/cupmb.git
cd cupmb
python3.8 -m pip install --upgrade --user -r requirements_examples.py
python3.8 setup.py install --user
python3.8 examples/0_simple.py
```
