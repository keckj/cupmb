import os, sys
import numpy as np
import cupy as cp

np.set_printoptions(threshold=sys.maxsize)

from pmb import logger
from pmb.cuda.backend import CudaBackend
from pmb.cuda.device import CudaDevice, CudaDeviceGroup

from pmb.utils.IO import cuda_kernels_dir
from pmb.utils.units import bytes2str


def extract_cuda_version(v):
    return f"{v // 1000}.{(v % 1000) // 10}"


logger.info("Loading CUDA backend.")

int_driver_version = cp.cuda.runtime.driverGetVersion()
int_cuda_version = cp.cuda.runtime.runtimeGetVersion()
driver_version = extract_cuda_version(int_driver_version)
cuda_version = extract_cuda_version(int_cuda_version)

if int_driver_version < 11000:
    msg = (
        "You are using an outdated NVIDIA driver, expected a CUDA >= 11.0 compatible driver but found version "
        f"{driver_version} ({int_driver_version}). Please upgrade your NVidia driver to the latest stable "
        "driver supported by your device at https://www.nvidia.fr/Download/index.aspx."
    )
    logger.warning(msg)

if int_cuda_version < 11000:
    msg = (
        "You are using an outdated CUDA version, expected at least CUDA 11.0 but found version "
        f"{cuda_version} ({int_cuda_version}). Please download and install lastest CUDA release from "
        "https://developer.nvidia.com/cuda-downloads and upgrade cupy."
    )
    logger.warning(msg)

msg = (
    f"Loaded cuda runtime {cuda_version} ({int_cuda_version}), "
    f"driver version is {driver_version} ({int_driver_version})."
)
logger.info(msg)

logger.info("Creating default CUDA device group containing all visible devices.")
device_count = CudaBackend.available_device_count()
if device_count <= 0:
    msg = "Could not find any CUDA device."
    logger.fatal(msg)
    raise RuntimeError(msg)
logger.info(f"Found {device_count} visible CUDA device(s).")

devices = CudaDeviceGroup()
for i in range(device_count):
    logger.debug(f"Initializing cuda device {i}...")
    devices.insert_device(i, CudaDevice(i))
    logger.info(f"Found cuda device {devices[i]}")

    attrs = devices[i].attrs
    logger.debug(f"Device UUID:                       {attrs.uuid.hex()}")
    logger.debug(
        "Compute capability:                "
        f"{'.'.join(map(str, devices[i].compute_capability))} (sm_{devices[i].architecture})"
    )
    logger.debug(f"Concurrent kernels:                {attrs.concurrent_kernels}")
    logger.debug(f"Async engine count:                {attrs.async_engine_count}")
    logger.debug(f"Multiprocessor count:              {attrs.multi_processor_count}")
    logger.debug(
        f"Global memory size:                {bytes2str(attrs.total_global_memory)}"
    )
    logger.debug(
        f"Constant memory size:              {bytes2str(attrs.total_constant_memory)}"
    )
    logger.debug(f"L2 cache size:                     {bytes2str(attrs.l2_cache_size)}")
    logger.debug(f"Warp size:                         {attrs.warp_size}")
    logger.debug(
        f"Max blocks per SM:                 {attrs.max_blocks_per_multi_processor}"
    )
    logger.debug(
        f"Max threads per SM:                {attrs.max_threads_per_multi_processor}"
    )
    logger.debug(f"Max threads per block:             {attrs.max_threads_per_block}")
    logger.debug(f"Max blockdim.x:                    {attrs.max_block_dim_x}")
    logger.debug(f"Max blockdim.y:                    {attrs.max_block_dim_y}")
    logger.debug(f"Max griddim.x:                     {attrs.max_grid_dim_x}")
    logger.debug(f"Max griddim.y:                     {attrs.max_grid_dim_y}")
    logger.debug(
        f"Max 32B registers per SM:          {attrs.max_registers_per_multiprocessor} "
        f"({bytes2str(32/8*attrs.max_registers_per_multiprocessor)})"
    )
    logger.debug(
        f"Max 32B registers per block:       {attrs.max_registers_per_block} "
        f"({bytes2str(32/8*attrs.max_registers_per_block)})"
    )
    logger.debug(
        f"Max shared memory per SM:          {bytes2str(attrs.max_shared_memory_per_multiprocessor)}"
    )
    logger.debug(
        f"Max shared memory per block:       {bytes2str(attrs.max_shared_memory_per_block)}"
    )
    logger.debug(
        f"Max shared memory per block optin: {bytes2str(attrs.max_shared_memory_per_block_optin)}"
    )

device_id = 0
logger.info(f"Using device {device_id}.")
device = devices[device_id]

dtype_to_ctype = {
    np.float32: "float",
    np.float64: "double",
}

source = os.path.join(cuda_kernels_dir, "pmb.cuh")
options = CudaBackend.default_build_options() + ("--restrict", f"-I{cuda_kernels_dir}")


line_table = np.asarray(
    [
        [-1, -1, -1, -1],  # case 0
        [0, 3, -1, -1],  # case 1
        [0, 1, -1, -1],  # case 2
        [3, 1, -1, -1],  # case 3
        [1, 2, -1, -1],  # case 4
        [0, 1, 3, 2],  # case 5
        [0, 2, -1, -1],  # case 6
        [3, 2, -1, -1],  # case 7
        [3, 2, -1, -1],  # case 8
        [0, 2, -1, -1],  # case 9
        [0, 3, 1, 2],  # case 10
        [1, 2, -1, -1],  # case 11
        [3, 1, -1, -1],  # case 12
        [0, 1, -1, -1],  # case 13
        [0, 3, -1, -1],  # case 14
        [-1, -1, -1, -1],  # case 15
    ],
    dtype=np.int8,
)

dtype = np.float64
ndim = 2
Nx = np.int32(50)
Ny = np.int32(50)
N = Nx * Ny

logger.info(f"Discretization is Nx={Nx} and Ny={Ny}")
Bx = np.uint32(32)
By = np.uint32(32)
B = Bx * By
logger.info(f"Block size is Bx={Bx} and By={By}")

Tx = np.uint32(128)

warpsize = device.attrs.warp_size
assert B % warpsize == 0, (Bx, warpsize)

Gx = (Nx + Bx - 2) // (Bx - 1)
Gy = (Ny + By - 2) // (By - 1)
Nblocks = Gx * Gy
Px = (Nblocks + Tx - 1) // Tx

isolevel = dtype(0.75)
x0 = dtype(0.0)
y0 = dtype(0.0)
Lx = dtype(1.0)
Ly = dtype(1.0)
dx = dtype(Lx / Nx)
dy = dtype(Ly / Ny)


def scalar(x, y):
    if np.isscalar(x):
        return np.sin(3 * np.pi * x, dtype=x.dtype) * np.cos(
            5 * np.pi * y, dtype=y.dtype
        )
    else:
        assert x.shape == y.shape, (x.shape, y.shape)
        inside_domain = np.logical_and(
            np.logical_and(x >= x0, x <= x0 + Lx), np.logical_and(y >= y0, y <= y0 + Ly)
        )
        output = np.empty_like(x * y)
        output[...] = np.where(
            inside_domain,
            np.sin(3 * np.pi * x, dtype=x.dtype) * np.cos(5 * np.pi * y, dtype=y.dtype),
            float("nan"),
        )
        return output


import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12, 6))
fig.suptitle(f"Marching squares: Nx={Nx}, Ny={Ny}, Bx={Bx}, By={By}, Gx={Gx}, Gy={Gy}")
axes[0].set_title("CPU")
axes[1].set_title("GPU")
dx = Lx / Nx
dy = Ly / Ny
x = x0 + np.arange(Nx + 1) * dx
y = y0 + np.arange(Ny + 1) * dy
x, y = np.meshgrid(x, y)
I = scalar(x, y)
for ax in axes:
    ax.imshow(
        I,
        vmin=-1.0,
        vmax=1.0,
        extent=[-dx / 2, Lx + dx / 2, -dy / 2, Ly + dy / 2],
        interpolation="none",
    )
    for i in range(0, Ny, By - 1):
        ax.axhline(y=y0 + i * dy, xmin=x0, xmax=x0 + Lx, zorder=100, color="r")
    ax.axhline(y=y0 + Ny * dy, xmin=x0, xmax=x0 + Lx, color="r")
    for i in range(0, Nx, Bx - 1):
        ax.axvline(x=x0 + i * dx, ymin=y0, ymax=y0 + Ly, zorder=100, color="r")
    for bid in range(Nblocks):
        bidx = bid % Gx
        bidy = bid // Gx
        ax.text(
            x=x0 + (bidx * (Bx - 1)) * dx,
            y=y0 + (bidy * (By - 1)) * dy,
            s=f"{bid}",
            zorder=1000,
        )
    ax.axvline(x=x0 + Nx * dx, ymin=y0, ymax=y0 + Ly, color="r")
fig.show()


h_minvalues = np.empty((Gy, Gx), dtype=dtype)
h_maxvalues = np.empty((Gy, Gx), dtype=dtype)

Ix = np.arange(Bx)
Iy = np.arange(By)
for i in range(Gy):
    for j in range(Gx):
        yidx = i * (By - 1) + Iy
        xidx = j * (Bx - 1) + Ix
        x = (x0 + xidx * dx).astype(dtype)
        y = (y0 + yidx * dy).astype(dtype)
        x, y = np.meshgrid(x, y)
        s = scalar(x, y) - isolevel
        h_minvalues[i, j] = np.nanmin(s)
        h_maxvalues[i, j] = np.nanmax(s)

h_block_mask = (
    (h_minvalues <= 0.0) & (h_maxvalues >= 0.0) & (h_minvalues < h_maxvalues)
).astype(np.uint8)
h_block_count = h_block_mask.sum().reshape((1,)).astype(np.uint32)
h_compacted_block_indices = (
    np.arange(Nblocks, dtype=np.int32)[h_block_mask.astype(np.bool_).ravel()]
).copy()
logger.info(f"Found {h_block_count.item()} active blocks containing solutions.")

vertices = []
for i in range(h_block_count.item()):
    bid = h_compacted_block_indices[i]
    bidy = bid // Gx
    bidx = bid % Gx
    yidx = bidy * (By - 1) + Iy
    xidx = bidx * (Bx - 1) + Ix
    x = (x0 + xidx * dx).astype(dtype)
    y = (y0 + yidx * dy).astype(dtype)
    x, y = np.meshgrid(x, y)
    s = scalar(x, y) - isolevel
    xx = x[:-1, :-1].ravel()
    yy = y[:-1, :-1].ravel()
    s0 = s[:-1, :-1].ravel()
    s1 = s[:-1, +1:].ravel()
    s2 = s[+1:, +1:].ravel()
    s3 = s[+1:, :-1].ravel()
    caseid = 1 * (s0 <= 0.0).astype(np.uint32)
    caseid += 2 * (s1 <= 0.0).astype(np.uint32)
    caseid += 4 * (s2 <= 0.0).astype(np.uint32)
    caseid += 8 * (s3 <= 0.0).astype(np.uint32)
    caseid[s0 != s0] = 0
    caseid[s1 != s1] = 0
    caseid[s2 != s2] = 0
    caseid[s3 != s3] = 0
    for j in range(caseid.size):
        config = line_table[caseid[j]]
        S = [s0[j], s1[j], s2[j], s3[j]]
        X = [xx[j], xx[j] + dx, xx[j] + dx, xx[j]]
        Y = [yy[j], yy[j], yy[j] + dy, yy[j] + dy]
        for k, e in enumerate(config):
            if e == -1:
                break
            i0 = (e + 0) % 4
            i1 = (e + 1) % 4
            W = S[i0] / (S[i0] - S[i1])
            xs = X[i0] + W * (X[i1] - X[i0])
            ys = Y[i0] + W * (Y[i1] - Y[i0])
            vertices.append((xs, ys))

h_vertices = np.asarray(tuple(sorted(vertices)), dtype=dtype)
h_vertex_and_index_count = np.asarray([h_vertices.size, 0], dtype=np.uint32)
X, Y = h_vertices[:, 0], h_vertices[:, 1]
axes[0].scatter(X, Y, zorder=10)
logger.info(
    f"Found {h_vertex_and_index_count[0]} python solutions: max(|f(x)|) = {abs(scalar(X,Y)).max()}"
)

d_block_mask = cp.empty(shape=h_block_mask.shape, dtype=h_block_mask.dtype)
d_block_count = cp.empty(shape=h_block_count.shape, dtype=h_block_count.dtype)
d_vertex_and_index_count = cp.empty(
    shape=h_vertex_and_index_count.shape, dtype=h_vertex_and_index_count.dtype
)
d_compacted_block_indices = cp.empty(
    shape=h_compacted_block_indices.shape,
    dtype=h_compacted_block_indices.dtype,
)
d_vertices = cp.empty(shape=h_vertices.shape, dtype=h_vertices.dtype)

ctype = dtype_to_ctype[dtype]
knames = (f"computeBlockMask{ndim}D<{ctype},{Bx}u,{By}u,{Nx}u,{Ny}u,{warpsize}u>",)
knames += (f"countActiveBlocks<{Tx}u,{Nblocks}u,{warpsize}u>",)
knames += (f"compactActiveBlocks<{Tx}u,{Nblocks}u,{warpsize}u>",)
knames += (f"computeSimplices{ndim}D<{ctype},{Bx}u,{By}u,{Nx}u,{Ny}u,{warpsize}u>",)
for k in knames:
    logger.info(k)
(
    compute_block_mask,
    count_active_blocks,
    compact_active_blocks,
    compute_simplices,
) = CudaBackend.build_module(
    knames,
    filepath=source,
    extract_kernels=True,
    options=options,
)

if B > compute_block_mask.max_threads_per_block:
    raise RuntimeError((B, compute_block_mask.max_threads_per_block))
if B > compute_simplices.max_threads_per_block:
    raise RuntimeError((B, compute_simplices.max_threads_per_block))
if Tx > count_active_blocks.max_threads_per_block:
    raise RuntimeError((Tx, count_active_blocks.max_threads_per_block))
if Tx > compact_active_blocks.max_threads_per_block:
    raise RuntimeError((Tx, compact_active_blocks.max_threads_per_block))

compute_block_mask((Gx, Gy), (Bx, By), (isolevel, x0, dx, y0, dy, d_block_mask))
block_mask = cp.asnumpy(d_block_mask)
logger.info("Block active mask:")
print("CPU block mask")
print(h_block_mask)
print("GPU block mask")
print(block_mask)
np.testing.assert_equal(block_mask, h_block_mask)

cp.cuda.runtime.memsetAsync(d_block_count.data.ptr, 0x0, d_block_count.nbytes, 0)
count_active_blocks((Px,), (Tx,), (d_block_mask, d_block_count))
block_count = cp.asnumpy(d_block_count)
logger.info(
    f"CPU counted {h_block_count.item()} active blocks and GPU {block_count.item()}."
)
np.testing.assert_equal(block_count, h_block_count)

cp.cuda.runtime.memsetAsync(d_block_count.data.ptr, 0x0, d_block_count.nbytes, 0)
compact_active_blocks(
    (Px,),
    (Tx,),
    (
        np.uint32(d_compacted_block_indices.size),
        d_block_mask,
        d_compacted_block_indices,
        d_block_count,
    ),
)
block_count_bis = cp.asnumpy(d_block_count)
compacted_block_indices = cp.asnumpy(d_compacted_block_indices)
logger.info(
    f"GPU successfully compacted {block_count_bis.item()} active block indices."
)
assert block_count == block_count_bis, (block_count, block_count_bis)
compacted_block_indices.sort()
print("CPU block indices")
print(h_compacted_block_indices)
print("GPU block indices")
print(compacted_block_indices)
np.testing.assert_equal(compacted_block_indices, h_compacted_block_indices)

logger.info("Generating triangles...")
cp.cuda.runtime.memsetAsync(
    d_vertex_and_index_count.data.ptr, 0x0, d_vertex_and_index_count.nbytes, 0
)
Nactive = block_count.item()
compute_simplices(
    (Nactive,),
    (Bx, By),
    (
        np.uint32(d_vertices.shape[0]),
        np.uint32(0),
        isolevel,
        x0,
        dx,
        y0,
        dy,
        d_compacted_block_indices,
        d_vertices,
        d_vertex_and_index_count,
    ),
)
vertex_and_index_count = cp.asnumpy(d_vertex_and_index_count)
vertices = cp.asnumpy(d_vertices)
print(vertices)
X, Y = vertices[:, 0], vertices[:, 1]
axes[1].scatter(X, Y, zorder=10)

try:
    np.testing.assert_equal(vertex_and_index_count, h_vertex_and_index_count)
    np.testing.assert_allclose(vertices, h_vertices)
    logger.info(
        f"Found {vertex_and_index_count[0]} grid-aligned zeros of function sin(3*π*x)*cos(5*π*x) on domain [0,1]."
    )
except:
    pass
plt.show()
