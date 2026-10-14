<!--
Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved

Project CUPMB and package cupmb are proprietary and confidential
and can not be copied, reproduced, modified and/or distributed without the express permission
of all copyright holders (Jean-Baptiste Keck).

Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
-->
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)]()
[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cuda 12](https://img.shields.io/badge/cuda-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit-archive)
[![Licence](https://img.shields.io/badge/license-proprietary-orange)](https://en.wikipedia.org/wiki/Proprietary_software)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Cuda Parallel Marching Blocks

A Cuda implementation of the Parallel Marching Blocks (PMB) algorithm as a Python3 module. If you are looking for an efficient marching squares / marching cubes implementation on the GPU, just give it a try. See [Parallel Marching Blocks: A Practical Isosurfacing Algorithm for Large Data on Many-Core Architectures](https://core.ac.uk/download/pdf/161941331.pdf) for more information about the algorithm.

## Requirements
- Linux, NVIDIA GPU and driver.
- Python >= 3.10.
- CUDA 12.x (tested with CUDA 12.9 on a sm_120 GPU). The CUDA 13 cupy wheel (`cupy-cuda13x`) can be selected with the `cupy13` extra.
- One of the two backends: [CuPy](https://cupy.dev/) (default) or [PyCUDA](https://documen.tician.de/pycuda/).
- Git submodules: [trove](https://github.com/bryancatanzaro/trove) and [cusvna](https://github.com/keckj/cusvna) (header only, shipped with the package).

## Features
- PMB outputs an indexed mesh: vertices are shared between the simplices of a block, no triangle soup as with other marching cubes algorithms. Vertices lying on the faces shared by two blocks are duplicated, weld them by position if you need a fully connected mesh.
- Compatible with both [CuPy](https://cupy.dev/) and [PyCUDA](https://documen.tician.de/pycuda/) backends and Cuda/OpenGL interoperability.
- 1D (zero crossings, single vertex simplices), 2D (marching squares, segments) and 3D (marching cubes, triangles) support, single or double precision for vertices. In 1D, `V` and `I` have shape `(n, 1)` and simplex `k` references vertex `k`, vertices are never duplicated on block boundaries.
- `uint32` or `uint64` simplex indices.
- Consistently oriented simplices: normals point from negative to positive scalar values (along the gradient). Triangles are counterclockwise when seen from the positive side (closed surfaces have a positive signed volume) and segments are counterclockwise around negative regions (positive signed area). 1D simplices (single vertices) carry no orientation, the normal is the sign of the derivative and can be computed as a vertex property.
- Scalar field can be analytic (function computed at runtime on the GPU) or stored in an array/texture. The sampled scalar field can be exported (`export_scalar_field=True`), it is stored x-fastest and indexed as `S[iz,iy,ix]` (shape `N[::-1]+1`, `S[iy,ix]` in 2D as expected by `matplotlib.imshow` or `contour`).
- Low memory consumption with user configurable allocation policies (and possibly user provided buffers).
- Handling of custom user defined vertex properties and user provided extra kernel parameters (constants or Python callables evaluated at each call).
- Support for dynamic domains (axis-aligned bounding box) and dynamic kernel parameters.
- Vertices and simplices are stored as array of structures, vector loads and stores are optimized with [trove](https://github.com/bryancatanzaro/trove).
- Support for runtime KD-tree-pyramid domain decomposition (binary tree in 1D, quadtree in 2D and octtree in 3D) from analytic block predicates (`pdef`). Predicates can be computed with range analysis by using [interval arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic) or [affine arithmetic](https://en.wikipedia.org/wiki/Affine_arithmetic), for instance with the [cusvna](https://github.com/keckj/cusvna) headers (`#include <cusvna/af1.cuh>`, requires `enable_cuda_std_lib=True`).

## How to get started
```
git clone --recursive https://gitlab.com/keckj/cupmb.git
cd cupmb
python3 -m venv .venv && source .venv/bin/activate
pip install ".[cupy12]"
python examples/0_simple.py
```
If the repository was cloned without `--recursive`, run `git submodule update --init --recursive` before installing.

Available extras (combine them with commas, for instance `pip install ".[cupy12,examples]"`):

| Extra      | Content                                                                    |
|------------|----------------------------------------------------------------------------|
| `cupy12`   | CuPy backend for CUDA 12 (`cupy-cuda12x`)                                  |
| `cupy13`   | CuPy backend for CUDA 13 (`cupy-cuda13x`)                                  |
| `pycuda`   | PyCUDA backend (`pycuda`, `cuda-python` for NVRTC)                         |
| `gl`       | Cuda/OpenGL interoperability (`cuda-python`, `PyOpenGL`)                   |
| `examples` | Dependencies of the examples (trimesh, open3d, matplotlib, glfw, ...)      |
| `test`     | Test suite (`pytest`, `scipy`)                                             |
| `dev`      | Test suite, `build`, `pre-commit`, `black`, `ruff`, `scikit-image`         |

All the dependencies of the examples (both backends included) can also be installed with `pip install -r requirements_examples.txt`.

## Minimal usage
```python
from cupmb import ParallelMarchingBlocks

fdef = r"""
namespace cupmb {
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i) {
        const T& x = trove::get<0>(p);
        const T& y = trove::get<1>(p);
        const T& z = trove::get<2>(p);
        return x*x + y*y + z*z - T(0.45)*T(0.45);
    }
}
"""
pmb = ParallelMarchingBlocks(3, fdef, N=64, x0=-0.5, L=1.0)
V, I = pmb(to_numpy=True)  # vertices (nv, 3) float32, triangles (nt, 3) uint32
```
Without `to_numpy=True` the output buffers stay on the device. When the zero level set is empty, `V` and `I` are `None`. Verbosity is controlled by the `CUPMB_LOGLEVEL` environment variable (`DEBUG`, `INFO`, `WARNING`, ...).

## Examples
- `0_simple.py` This simple example will show you how to execute the marching cubes algorithm to find the zero levelset of a scalar field representing a sphere. The scalar field is defined as $\phi(x,y,z) = x^2 + y^2 + z^2 - r^2$ where $r=0.45$ is a constant. The function is analytic and implemented directly as a device function. The default bounding box is $[0,1]^3$ so we offset it by $x_0=-0.5$ to find the zeros for $(x,y,z) \in [-0.5,+0.5]^3$. The output mesh (vertices `V`, simplices `I`) is copied to the host (numpy arrays) and is then displayed by using the [trimesh](https://trimesh.org/) module. Parameter `N` controls the resolution of the marching cubes algorithm.
- `1_animation.py` Sphere with a radius given as an extra kernel parameter (`extra_params=[("r", "T")]`, passed as `pmb(r=...)`), the mesh is recomputed at each frame and displayed with [open3d](https://www.open3d.org/).
- `2_normals.py` User defined vertex properties (`VertexProperty`): normals are computed on the GPU alongside the vertices and used for shading in an open3d GUI.
- `3_glinterop_cupy.py`, `3_glinterop_pycuda.py` Cuda/OpenGL interoperability: vertices and simplices are written directly into OpenGL buffers (no host round trip) and rendered with glfw/PyOpenGL (`lock_buffer`/`replace_buffer`), with the CuPy and PyCUDA backends respectively.
- `4_isolines.py` Marching squares in 2D: isolines of an analytic scalar field, the sampled scalar field is exported (`export_scalar_field=True`) and displayed with matplotlib, the cupmb segments (red) are compared to `matplotlib.contour` (black).
- `5_moving_sphere.py` Extra kernel parameters (`ExtraParameter`) given as Python callables that are evaluated before each call, here the center of a moving sphere.
- `6_moving_bbox.py` Dynamic domain: the bounding box follows the moving sphere through a user provided `update_bbox` callback.
- `7_predicate.py` Benchmark of the block predicate based domain decomposition against the naive block extraction, the predicate is written with cusvna affine arithmetic. Run `python examples/7_predicate.py --help` for the options (`-N` grid size, `--rigor` autotuning rigor, `--runs`), large grids require a lot of device memory. The mesh is then displayed with open3d at a lower resolution (`--display-N`, default 512, `--no-display` to disable), shaded with normals computed on the GPU as a vertex property.
- `8_levelset.py` Real time meshing of the animated scene of the last cusvna example (gyroid ball surrounded by a ring of heart, Kummer, Barth, tetrahedral, twisted box and metaballs level sets): the shapes are written once as templates evaluated on floats for the scalar field and on cusvna affine forms for the block predicate, vertices, triangles, normals and colors are written directly into OpenGL buffers and rendered with glfw/PyOpenGL. Controls (shown in an overlay when [pyimgui](https://github.com/pyimgui/pyimgui) is installed, logged at startup otherwise): `SPACE` pause, `+`/`-` (or `UP`/`DOWN`) animation speed, `*`/`/` (or `PAGE UP`/`PAGE DOWN`) multiply or divide `N` by 2 between 64 and 5120 (the kernels are rebuilt with the `ESTIMATE` autotuning rigor), `W` wireframe, mouse to rotate and zoom, `Q`/`ESC` quit. Options: `-N` grid size (default 2560), `--no-predicate`, `--rigor` autotuning rigor, `--frames` number of frames, `--no-display` headless benchmark with and without predicate (median meshing times, triangle counts are checked to be identical). The predicate scales with the surface instead of the volume, it is slower at small `N` and pays off at high resolution.

## Testing
```
pip install ".[cupy12,test]"
pytest                 # requires a CUDA device, GPU tests are skipped otherwise
pytest --runslow       # also runs the large grid and large index tests (several GB of device memory)
```
If `scikit-image` is installed (`dev` extra), the 3D meshes are also cross-checked against `skimage.measure.marching_cubes`.

## Development
```
pip install -e ".[cupy12,dev]"
pre-commit install
python -m build        # sdist and wheel in dist/
```
