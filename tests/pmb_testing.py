#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
"""
Shared helpers for the cupmb test suite: analytic scalar fields (device source code and host reference),
mesh welding and geometric/topological checks.
"""

import os, sys, subprocess

import numpy as np

AXES = ("x", "y", "z")


def scalar_source(dim, body, extra_args=""):
    """
    Device source code of cupmb::scalar for a field given as a C++ expression `body` of the
    coordinates x, y (and z) and of the grid indices ix, iy (and iz).
    """
    coords = "\n".join(f"        const T {a} = trove::get<{k}>(p);" for (k, a) in enumerate(AXES[:dim]))
    indices = "\n".join(f"        const I i{a} = trove::get<{k}>(i);" for (k, a) in enumerate(AXES[:dim]))
    return rf"""
namespace cupmb {{
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i{extra_args}) {{
{coords}
{indices}
        (void)p; (void)i;
        return {body};
    }}
}}
"""


def predicate_source(dim, body, includes=()):
    """
    Device source code of the cupmb::may_contain_zeros block predicate. `body` is a C++ statement
    block that can use `domain.a[k]` and `domain.b[k]` (k=0 is x) and must return a bool.
    """
    headers = "\n".join(f"#include <{h}>" for h in includes)
    return rf"""
{headers}
namespace cupmb {{
    template <typename T, typename I, I dim>
    __device__ __forceinline__ bool may_contain_zeros(const BoundingBox<T,dim>& domain) {{
        (void)domain;
        {body}
    }}
}}
"""


ALWAYS_TRUE_PREDICATE = "return true;"
ALWAYS_FALSE_PREDICATE = "return false;"


def sphere_predicate_body(center, radius):
    """
    Conservative interval test for the sphere |p - center| = radius, written without cusvna:
    returns false only if the block provably does not intersect the sphere.
    """
    dim = len(center)
    lines = ["T lo = T(0), hi = T(0);"]
    for k, c in enumerate(center):
        lines.append(
            f"{{ const T a = domain.a[{k}] - T({float(c)!r}), b = domain.b[{k}] - T({float(c)!r}); "
            "const T aa = a*a, bb = b*b; "
            "lo += ((a <= T(0)) && (b >= T(0))) ? T(0) : (aa < bb ? aa : bb); "
            "hi += (aa > bb ? aa : bb); }"
        )
    r2 = float(radius) ** 2
    lines.append(f"const T r2 = T({r2!r});")
    lines.append("const T eps = T(1e-3);")
    lines.append("return (lo <= r2 + eps) && (hi >= r2 - eps);")
    assert dim in (2, 3)
    return "\n        ".join(lines)


class Field:
    """
    Analytic scalar field with a device implementation (C++ expression) and a host reference.

    hessian_bound: upper bound of |d^2 f / dx_k^2| over the domain (marching cubes linear interpolation
                   error on f along an edge is bounded by hessian_bound * dx^2 / 8).
    gradient_bound: upper bound of |grad f| over the domain (used for rounding error tolerances).
    """

    def __init__(self, name, dim, body, f, hessian_bound, gradient_bound, distance=None, closed=False):
        self.name = name
        self.dim = dim
        self.body = body
        self.f = f
        self.hessian_bound = hessian_bound
        self.gradient_bound = gradient_bound
        self.distance = distance
        self.closed = closed

    @property
    def fdef(self):
        return scalar_source(self.dim, self.body)

    def __repr__(self):
        return f"{self.name}{self.dim}D"


def sphere(dim, center=(0.041, -0.027, 0.013), radius=0.31):
    c = np.asarray(center[:dim], dtype=np.float64)
    terms = " + ".join(f"({a} - T({float(ck)!r}))*({a} - T({float(ck)!r}))" for (a, ck) in zip(AXES, c))
    body = f"{terms} - T({float(radius)!r})*T({float(radius)!r})"

    def f(P):
        return ((P - c) ** 2).sum(axis=-1) - radius**2

    def distance(P):
        return np.abs(np.linalg.norm(P - c, axis=-1) - radius)

    field = Field("sphere" if dim == 3 else "circle", dim, body, f, 2.0, 2.0, distance, closed=True)
    field.center = c
    field.radius = radius
    return field


def plane(dim, normal=(1.0, 0.37, -0.21), offset=0.0123):
    n = np.asarray(normal[:dim], dtype=np.float64)
    n /= np.linalg.norm(n)
    terms = " + ".join(f"T({float(nk)!r})*{a}" for (a, nk) in zip(AXES, n))
    body = f"{terms} - T({float(offset)!r})"

    def f(P):
        return P @ n - offset

    def distance(P):
        return np.abs(f(P))

    return Field("plane" if dim == 3 else "line", dim, body, f, 0.0, 1.0, distance)


def sines(dim):
    if dim == 2:
        body = "sin(T(5)*x)*cos(T(4)*y) + T(0.3)*sin(T(6)*y + T(0.5)) + T(0.1)"

        def f(P):
            x, y = P[..., 0], P[..., 1]
            return np.sin(5 * x) * np.cos(4 * y) + 0.3 * np.sin(6 * y + 0.5) + 0.1

        hessian_bound, gradient_bound = 25.0 + 16.0 + 10.8, 5.0 + 4.0 + 1.8
    else:
        body = "sin(T(5)*x)*cos(T(4)*y) + T(0.3)*sin(T(6)*z + T(0.5)) + T(0.1)"

        def f(P):
            x, y, z = P[..., 0], P[..., 1], P[..., 2]
            return np.sin(5 * x) * np.cos(4 * y) + 0.3 * np.sin(6 * z + 0.5) + 0.1

        hessian_bound, gradient_bound = 25.0, 5.0 + 4.0 + 1.8
    return Field("sines", dim, body, f, hessian_bound, gradient_bound)


FIELDS = {"sphere": sphere, "plane": plane, "sines": sines}


def grid_coordinates(pmb):
    """1D host grid coordinates along each axis (N+1 points per axis)."""
    x0 = np.asarray(pmb.x0, dtype=np.float64)
    dx = np.asarray(pmb.dx, dtype=np.float64)
    N = np.asarray(pmb.N, dtype=np.int64)
    return tuple(x0[k] + dx[k] * np.arange(N[k] + 1) for k in range(len(N)))


def sample_field(pmb, f):
    """Sample f on the host grid, result is indexed [ix, iy, (iz)]."""
    coords = np.meshgrid(*grid_coordinates(pmb), indexing="ij")
    P = np.stack(coords, axis=-1)
    return f(P)


def as_numpy(a):
    if a is None:
        return None
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def weld(V, I):
    """
    Merge vertices with identical positions (vertices are shared inside a block but duplicated on block faces).
    Returns welded vertices and simplices with remapped indices.
    """
    Vu, inverse = np.unique(V, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return Vu, inverse[np.asarray(I, dtype=np.int64)]


def edges_of(I):
    """Undirected edges of a triangle (or segment) mesh."""
    I = np.asarray(I, dtype=np.int64)
    if I.shape[1] == 2:
        return np.sort(I, axis=1)
    E = np.concatenate([I[:, [0, 1]], I[:, [1, 2]], I[:, [2, 0]]], axis=0)
    return np.sort(E, axis=1)


def non_degenerate(I):
    I = np.asarray(I)
    if I.shape[1] == 2:
        return I[I[:, 0] != I[:, 1]]
    keep = (I[:, 0] != I[:, 1]) & (I[:, 1] != I[:, 2]) & (I[:, 2] != I[:, 0])
    return I[keep]


def signed_volume(V, I):
    V = np.asarray(V, dtype=np.float64)
    a, b, c = V[I[:, 0]], V[I[:, 1]], V[I[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def assert_same_points(A, B, atol):
    """A and B contain the same points up to atol (any order, duplicates ignored)."""
    from scipy.spatial import cKDTree

    A = np.unique(np.asarray(A, dtype=np.float64), axis=0)
    B = np.unique(np.asarray(B, dtype=np.float64), axis=0)
    dAB, _ = cKDTree(B).query(A)
    dBA, _ = cKDTree(A).query(B)
    assert dAB.max() <= atol and dBA.max() <= atol, (dAB.max(), dBA.max(), atol)


def ftype_eps(ftype):
    return float(np.finfo(ftype).eps)


def run_python(code, env=None, timeout=300):
    """
    Run a Python snippet in a fresh interpreter (own CUDA context), with this directory and the
    directory containing the tested cupmb package in sys.path. Returns the completed process.
    """
    import cupmb

    paths = [os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.dirname(cupmb.__file__))]
    full_env = dict(os.environ)
    full_env.update(env or {})
    full_env["PYTHONPATH"] = os.pathsep.join(paths + [full_env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(
        [sys.executable, "-c", code], env=full_env, capture_output=True, text=True, timeout=timeout, cwd=paths[0]
    )
