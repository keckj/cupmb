import os, sys, re
import sympy as sm
import numpy as np
import cupy as cp

import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

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
itype_to_ctype = {
    np.uint32: "unsigned int",
    np.uint64: "unsigned long long int",
}

source = os.path.join(cuda_kernels_dir, "pmb.cuh")
options = CudaBackend.default_build_options() + ("--restrict", f"-I{cuda_kernels_dir}")


line_table = np.asarray(
    [
        [-1, -1, -1, -1],  # case 0
        [+0, +3, -1, -1],  # case 1
        [+0, +1, -1, -1],  # case 2
        [+3, +1, -1, -1],  # case 3
        [+1, +2, -1, -1],  # case 4
        [+0, +1, +3, +2],  # case 5
        [+0, +2, -1, -1],  # case 6
        [+3, +2, -1, -1],  # case 7
        [+3, +2, -1, -1],  # case 8
        [+0, +2, -1, -1],  # case 9
        [+0, +3, +1, +2],  # case 10
        [+1, +2, -1, -1],  # case 11
        [+3, +1, -1, -1],  # case 12
        [+0, +1, -1, -1],  # case 13
        [+0, +3, -1, -1],  # case 14
        [-1, -1, -1, -1],  # case 15
    ],
    dtype=np.int8,
)

local_vertex_count = np.asarray(
    [int(0 in xi) + int(3 in xi) for xi in line_table], dtype=np.uint8
)
print("Vertex count: ", local_vertex_count)

edges_count = np.asarray([(xi != -1).sum() // 2 for xi in line_table], dtype=np.uint8)
print("Edge count:   ", edges_count)

generate_v0 = np.asarray([0 in xi for xi in line_table], dtype=np.uint8)
generate_v3 = np.asarray([3 in xi for xi in line_table], dtype=np.uint8)


def boolean_expression(
    values, nbits=None, ignore=None, prefix=None, postfix=None, subs=None
):
    nbits = np.iinfo(values.dtype).bits if (nbits is None) else nbits
    assert nbits > 0, nbits

    ignore = set() if (ignore is None) else set(ignore)
    prefix = "" if (prefix is None) else prefix
    postfix = "" if (postfix is None) else postfix
    subs = (lambda x: x) if (subs is None) else subs

    expr = ""
    written = False
    for i in range(nbits):
        M = np.full_like(values, dtype=bool, fill_value=False)
        for ignored_value in ignore:
            M |= values == ignored_value
        (U,) = np.where(M)
        (I,) = np.where(((values >> (nbits - 1 - i)) & 1) & (~M))
        sop = sm.SOPform(ss, I.tolist(), U.tolist())
        if sop == False:  # noqa: E712
            continue
        sops = str(sop).replace("~", "!").replace("&", "&&").replace("|", "||")
        if written:
            expr += " + "
        written = True
        if nbits - 1 - i == 0:
            if sop == True:  # noqa: E712
                expr += "1"
            else:
                expr += f"({sops})"
        else:
            if sop == True:  # noqa: E712
                expr += str(1 << (nbits - 1 - i))
            else:
                expr += f"(({sops}) << {nbits-1-i})"
    if not written:
        expr += "0"
    return subs(prefix + expr + postfix)


ss = sm.symbols("b3 b2 b1 b0")


def subs(x):
    return re.sub(r"b(\d)", r"b[\1]", x)


print(
    boolean_expression(
        local_vertex_count,
        nbits=2,
        prefix="unsigned char vertex_count = ",
        postfix=";",
        subs=subs,
    )
)
print(
    boolean_expression(
        edges_count,
        nbits=2,
        prefix="unsigned char edge_count = ",
        postfix=";",
        subs=subs,
    )
)
print(
    boolean_expression(
        generate_v0,
        nbits=2,
        prefix="unsigned char generate_v0 = ",
        postfix=";",
        subs=subs,
    )
)
print(
    boolean_expression(
        generate_v3,
        nbits=2,
        prefix="unsigned char generate_v3 = ",
        postfix=";",
        subs=subs,
    )
)
print("unsigned char edge_ids[4];")
for k in range(4):
    lt = line_table[:, k].copy()
    print(
        boolean_expression(
            lt, nbits=4, ignore={-1}, prefix=f"edge_ids[{k}] = ", postfix=";", subs=subs
        )
    )


def required_shared_memory_bytes(device, B, dtype, itype):
    warpsize = device.attrs.warp_size
    assert B % warpsize == 0, B
    assert B // warpsize <= warpsize, (B, warpsize)
    warps_per_block = B // warpsize
    m0 = B * dtype(0).itemsize
    m1 = B * (itype(0).itemsize + np.uint8(0).itemsize)
    m2 = 2 * (warps_per_block + 1) * itype(0).itemsize
    return max(m0, m1, m2)


dtype = np.float32
itype = np.uint32
ndim = 2
Nx = itype(16)
Ny = itype(16)
N = Nx * Ny

logger.info(f"Discretization is Nx={Nx} and Ny={Ny}")
Bx = np.uint32(32)
By = np.uint32(32)
B = Bx * By
shared_memory_bytes = required_shared_memory_bytes(device, B, dtype, itype)
logger.info(
    f"Block size is Bx={Bx} and By={By}, requires {shared_memory_bytes//1024}kiB shared memory"
)

Tx = np.uint32(128)

warpsize = device.attrs.warp_size

Gx = (Nx + Bx - 2) // (Bx - 1)
Gy = (Ny + By - 2) // (By - 1)
Nblocks = Gx * Gy
Px = (Nblocks + Tx - 1) // Tx

isolevel = dtype(-0.75)
x0 = dtype(0)
y0 = dtype(0)
Lx = dtype(1)
Ly = dtype(1)
dx = dtype(Lx / Nx)
dy = dtype(Ly / Ny)


def scalar(x, y):
    if np.isscalar(x):
        return np.sin(2 * np.pi * x, dtype=x.dtype) * np.cos(
            2 * np.pi * y, dtype=y.dtype
        )
    else:
        assert x.shape == y.shape, (x.shape, y.shape)
        inside_domain = np.logical_and(
            np.logical_and(x >= x0, x <= x0 + Lx), np.logical_and(y >= y0, y <= y0 + Ly)
        )
        output = np.empty_like(x * y)
        output[...] = np.where(
            inside_domain,
            np.sin(2 * np.pi * x, dtype=x.dtype) * np.cos(2 * np.pi * y, dtype=y.dtype),
            float("nan"),
        )
        return output


fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12, 6))
fig.suptitle(f"Marching squares: Nx={Nx}, Ny={Ny}, Bx={Bx}, By={By}, Gx={Gx}, Gy={Gy}")
axes[0].set_title("CPU")
axes[1].set_title("GPU")
x = x0 + np.arange(Nx + 1) * dx
y = y0 + np.arange(Ny + 1) * dy
x, y = np.meshgrid(x, y)
I = scalar(x, y)
E = (I - isolevel) <= 0
C = np.where(np.isnan(I), "g", np.where(E, "m", "b"))
for ax in axes:
    ax.imshow(
        I,
        vmin=-1.0,
        vmax=1.0,
        extent=[-dx / 2, Lx + dx / 2, -dy / 2, Ly + dy / 2],
        interpolation="none",
    )
    ax.scatter(x.ravel(), y.ravel(), c=C.ravel(), s=10, zorder=100000)
    for i in range(0, Ny + 1):
        if (i % (By - 1) == 0) or (i == Ny):
            ax.axhline(y=y0 + i * dy, xmin=x0, xmax=x0 + Lx, zorder=100, color="r")
        else:
            ax.axhline(
                y=y0 + i * dy,
                xmin=x0,
                xmax=x0 + Lx,
                zorder=100,
                color="k",
                alpha=1.0,
                ls="--",
                lw=1,
            )
    for i in range(0, Nx + 1):
        if (i % (Bx - 1) == 0) or (i == Nx):
            ax.axvline(x=x0 + i * dx, ymin=y0, ymax=y0 + Ly, zorder=100, color="r")
        else:
            ax.axvline(
                x=x0 + i * dx,
                ymin=y0,
                ymax=y0 + Ly,
                zorder=100,
                color="k",
                alpha=1.0,
                ls="--",
                lw=1,
            )
    for bid in range(Nblocks):
        bidx = bid % Gx
        bidy = bid // Gx
        ax.text(
            x=x0 + (bidx * (Bx - 1)) * dx + 0.1 * dx,
            y=y0 + (bidy * (By - 1)) * dy + 0.1 * dy,
            s=f"{bid}",
            zorder=1000,
            size=18,
        )
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
h_block_count = h_block_mask.sum().reshape((1,)).astype(itype)
h_compacted_block_indices = (
    np.arange(Nblocks, dtype=itype)[h_block_mask.astype(np.bool_).ravel()]
).copy()
logger.info(f"Found {h_block_count.item()} active blocks containing solutions.")

vertices = []
edges = []
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
    xx = x.ravel()
    yy = y.ravel()
    s0, s1, s2, s3 = (np.full_like(s, fill_value=np.nan) for i in range(4))
    s0[...] = s
    s1[:, :-1] = s[:, +1:]
    s2[:-1, :-1] = s[+1:, +1:]
    s3[:-1, :] = s[+1:, :]
    s0, s1, s2, s3 = map(np.ravel, (s0, s1, s2, s3))
    caseid = 1 * (s0 <= 0.0).astype(np.uint8)
    caseid += 2 * (s1 <= 0.0).astype(np.uint8)
    caseid += 4 * (s2 <= 0.0).astype(np.uint8)
    caseid += 8 * (s3 <= 0.0).astype(np.uint8)

    thread_first_vertex_id = []
    thread_nvertices = []

    # generate vertices
    for j in range(caseid.size):
        thread_first_vertex_id.append(len(vertices))
        config = line_table[caseid[j]]
        S = [s0[j], s1[j], np.nan, s3[j]]
        X = [xx[j], xx[j] + dx, xx[j] + dx, xx[j]]
        Y = [yy[j], yy[j], yy[j] + dy, yy[j] + dy]
        local_idx = 0
        for k, e in enumerate(config):
            if e == -1:
                break
            if e in {1, 2}:
                continue
            i0 = (e + 0) % 4
            i1 = (e + 1) % 4
            W = S[i0] / (S[i0] - S[i1])
            xs = X[i0] + W * (X[i1] - X[i0])
            ys = Y[i0] + W * (Y[i1] - Y[i0])
            vertices.append((xs, ys))
            local_idx += 1
        thread_nvertices.append(local_idx)

    # generate edges
    for j in range(caseid.size):
        if j % Bx == 0:
            continue
        case_id = caseid[j]
        config = line_table[caseid[j]]
        local_indices = []
        local_idx = 0
        for k, e in enumerate(config):
            if e == -1:
                break
            if e == 1:
                if thread_nvertices[j + 1] == 0:
                    print(f"ERROR: cid={case_id:2d}, e=1, i={j//Bx:2d}, j={j%Bx:2d}")
                idx = thread_first_vertex_id[j + 1] + (thread_nvertices[j + 1] > 1)
            elif e == 2:
                if thread_nvertices[j + Bx] == 0:
                    print(f"ERROR: cid={case_id:2d}, e=2, i={j//Bx:2d}, j={j%Bx:2d}")
                idx = thread_first_vertex_id[j + Bx]
            else:
                idx = thread_first_vertex_id[j] + local_idx
                local_idx += 1
            local_indices.append(idx)
        edges.extend(local_indices)

h_vertices = np.asarray(vertices, dtype=dtype)
h_edges = np.asarray(edges, dtype=itype).reshape(-1, 2)
h_vertex_and_edge_count = np.asarray(
    [h_vertices.shape[0], h_edges.shape[0]], dtype=itype
)

X, Y = h_vertices[:, 0], h_vertices[:, 1]
lines = [[tuple(h_vertices[j]) for j in i] for i in h_edges]
lc = matplotlib.collections.LineCollection(lines)
axes[0].scatter(X, Y, zorder=10, c="y", marker="x", s=80)
axes[0].add_collection(lc)
logger.info(
    f"Found {h_vertex_and_edge_count[0]} python solutions: max(|f(x)|) = {abs(scalar(X,Y)).max()}"
)

assert h_block_mask.dtype == np.uint8
assert h_block_count.dtype == itype
assert h_vertices.dtype == dtype
assert h_edges.dtype == itype
assert h_vertex_and_edge_count.dtype == itype

d_block_mask = cp.empty(shape=h_block_mask.shape, dtype=h_block_mask.dtype)
d_block_count = cp.empty(shape=h_block_count.shape, dtype=h_block_count.dtype)
d_vertex_and_edge_count = cp.empty(
    shape=h_vertex_and_edge_count.shape, dtype=h_vertex_and_edge_count.dtype
)
d_compacted_block_indices = cp.empty(
    shape=h_compacted_block_indices.shape,
    dtype=h_compacted_block_indices.dtype,
)
d_vertices = cp.empty(shape=h_vertices.shape, dtype=h_vertices.dtype)
d_edges = cp.empty(shape=h_edges.shape, dtype=h_edges.dtype)

flt_type = dtype_to_ctype[dtype]
int_type = itype_to_ctype[itype]
knames = (
    f"computeBlockMask{ndim}D<{flt_type},{int_type},{Bx}u,{By}u,{Nx}u,{Ny}u,{warpsize}u>",
)
knames += (f"countActiveBlocks<{int_type},{Nblocks}u,{Tx}u,{warpsize}u>",)
knames += (f"compactActiveBlocks<{int_type},{Nblocks}u,{Tx}u,{warpsize}u>",)
knames += (
    f"computeSimplices{ndim}D<{flt_type},{int_type},{Bx}u,{By}u,{Nx}u,{Ny}u,{warpsize}u>",
)
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
compute_simplices.max_dynamic_shared_size_bytes = shared_memory_bytes

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
        itype(d_compacted_block_indices.size),
        d_block_mask,
        d_compacted_block_indices,
        d_block_count,
    ),
)
block_count_bis = cp.asnumpy(d_block_count)
compacted_block_indices = cp.asnumpy(d_compacted_block_indices)
logger.info(f"GPU successfully compacted {block_count_bis.item()} active block edges.")
assert block_count == block_count_bis, (block_count, block_count_bis)
compacted_block_indices.sort()
print("CPU block edges")
print(h_compacted_block_indices)
print("GPU block edges")
print(compacted_block_indices)
np.testing.assert_equal(compacted_block_indices, h_compacted_block_indices)

logger.info("Generating triangles...")
cp.cuda.runtime.memsetAsync(
    d_vertex_and_edge_count.data.ptr, 0x0, d_vertex_and_edge_count.nbytes, 0
)
Nactive = block_count.item()
compute_simplices(
    (Nactive,),
    (Bx, By),
    (
        itype(d_vertices.shape[0]),
        itype(d_edges.shape[0]),
        isolevel,
        x0,
        dx,
        y0,
        dy,
        d_compacted_block_indices,
        d_vertices,
        d_edges,
        d_vertex_and_edge_count,
    ),
    shared_mem=shared_memory_bytes,
)
vertex_and_edge_count = cp.asnumpy(d_vertex_and_edge_count)
vertices = cp.asnumpy(d_vertices)
edges = cp.asnumpy(d_edges)
X, Y = vertices[:, 0], vertices[:, 1]
lines = [[tuple(vertices[j]) for j in i] for i in edges]
lc = matplotlib.collections.LineCollection(lines)
axes[1].scatter(X, Y, zorder=10, c="y", marker="x", s=80)
axes[1].add_collection(lc)

logger.info(
    f"GPU found {vertex_and_edge_count[0]} grid-aligned zeros "
    f"on domain [0,1] while CPU found {h_vertex_and_edge_count[0]}."
)
logger.info(
    f"GPU found {vertex_and_edge_count[1]} vertex edges to represent isolines "
    f"while CPU found {h_vertex_and_edge_count[1]}."
)

try:
    np.testing.assert_equal(vertex_and_edge_count, h_vertex_and_edge_count)
    np.testing.assert_allclose(vertices, h_vertices)
except:
    pass
plt.show()
