#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
# Real time meshing of an animated scene with a block predicate.
#
# The scene is the one of the last cusvna example (ext/cusvna/examples/05_levelset.py and
# ext/cusvna/examples/cuda/levelset.cu): a gyroid ball surrounded by a ring of animated level sets
# (heart, Kummer and Barth surfaces, tetrahedral quartic, twisted box and metaballs), standing above
# a checkered floor. There the level sets are ray marched with OptiX and cusvna interval marching,
# here their zero level set is extracted by cupmb at every frame and rendered with OpenGL.
#
# The shapes are written once as templates on the evaluation type: they are evaluated on floats for
# the scalar field and on cusvna affine forms cusvna::AF1<float,3> for the block predicate, which
# discards the empty blocks of the octree before any scalar is evaluated. Each object lives in its
# own bounding box, the scalar field is the union (min) of the objects and is positive outside of
# all the bounding boxes. The floor is not meshed, it is drawn as a plane by the viewer.
#
# Vertices, triangles, normals and colors (vertex properties) are written directly into OpenGL
# buffers (Cuda/OpenGL interoperability, see 3_glinterop_cupy.py). Run with --no-display for a
# headless benchmark of the meshing with and without predicate.
#
# Viewer controls: SPACE pause, +/- animation speed, * and / to multiply or divide the resolution by two
# (the number of cells is a power of two on each axis, kernels are compiled once per resolution and then
# loaded from the kernel cache), W wireframe, Q or ESC to quit. They are shown in an overlay if pyimgui is
# installed.
#
# The cost of the predicate scales with the surface (N^2 blocks are refined) while the cost of the
# naive approach scales with the volume (N^3 cells are classified): the predicate is slower at small
# resolutions (N < 2048 on a RTX 5090, where meshing takes a few milliseconds anyway) and pays off at
# high resolutions (1.5x faster at N=2560, 2.6x at N=4096).
#
import os
import sys
import enum
import ctypes
import argparse

try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 12.6
except ImportError:
    from cuda import cudart  # older cuda-python

if "CUPMB_LOGLEVEL" not in os.environ:
    os.environ["CUPMB_LOGLEVEL"] = "INFO"

import numpy as np
import cupy as cp

from cupmb import (
    ParallelMarchingBlocks,
    Backend,
    BufferType,
    VertexProperty,
    AutotuningRigor,
    logger,
)
from cupmb.utils.units import time2str


# ---------------------------------------------------------------------------------------------------
# Scene (same objects and animation as make_scene() in ext/cusvna/examples/05_levelset.py)
# ---------------------------------------------------------------------------------------------------
class Shape(enum.IntEnum):
    """Must match the Shape enum of the device code below (same values as cusvna's levelset.cu)."""

    GYROID = 1
    METABALLS = 2
    TWISTED_BOX = 3
    KUMMER = 4
    BARTH = 5
    HEART = 6
    TETRAHEDRAL = 7


ALBEDO = {
    Shape.GYROID: (1.00, 0.55, 0.10),
    Shape.METABALLS: (0.90, 0.15, 0.45),
    Shape.TWISTED_BOX: (0.05, 0.65, 0.60),
    Shape.KUMMER: (0.20, 0.35, 0.95),
    Shape.BARTH: (0.45, 0.80, 0.15),
    Shape.HEART: (0.85, 0.05, 0.08),
    Shape.TETRAHEDRAL: (0.55, 0.25, 0.85),
}

FLOOR_Y = -1.5


def make_scene():
    """A gyroid ball surrounded by a ring of animated level sets: (center, scale, spin, shape, phase)."""
    objects = [((0.0, FLOOR_Y + 1.45, 0.0), 1.4, 0.25, Shape.GYROID, 0.0)]
    # around the gyroid, starting in front of the camera
    ring = [Shape.HEART, Shape.KUMMER, Shape.BARTH, Shape.TETRAHEDRAL, Shape.TWISTED_BOX, Shape.METABALLS]
    ring_radius, ring_scale = 4.0, 0.95
    for i, shape in enumerate(ring):
        angle = np.pi / 2 + 2 * np.pi * (i + 0.5) / len(ring)
        center = (ring_radius * np.cos(angle), FLOOR_Y + ring_scale + 0.1, -ring_radius * np.sin(angle))
        objects.append((center, ring_scale, 0.5 * (-1) ** i, shape, 1.7 * i))
    return objects


def f32(x):
    """Float literal that is parsed to the exact same float32 value on the device."""
    return f"{float(np.float32(x))!r}f"


def make_object_table(objects):
    """
    Device table of the objects. The shape of an object lives in [-scale, scale]^3 around its
    center whatever its rotation around the y axis: lo/hi is its world bounding box, computed once
    here in single precision so that the scalar field and the predicate test the exact same box.
    """
    rows = []
    for center, scale, spin, shape, phase in objects:
        c = np.asarray(center, dtype=np.float32)
        lo, hi = c - np.float32(scale), c + np.float32(scale)
        rows.append(
            "        {"
            + f"{{{', '.join(map(f32, c))}}}, {f32(scale)}, {f32(spin)}, {f32(phase)}, {shape.name}, "
            + f"{{{', '.join(map(f32, lo))}}}, {{{', '.join(map(f32, hi))}}}, "
            + f"{{{', '.join(map(f32, ALBEDO[shape]))}}}"
            + "},"
        )
    rows = "\n".join(rows)
    return rf"""
namespace levelset {{
    constexpr int object_count = {len(objects)};
    __constant__ Object objects[object_count] = {{
{rows}
    }};
}}
"""


def scene_domain(objects, margin=0.02):
    """Bounding box of all the objects (the floor is not meshed)."""
    lo = np.min([np.asarray(c) - s for (c, s, *_) in objects], axis=0) - margin
    hi = np.max([np.asarray(c) + s for (c, s, *_) in objects], axis=0) + margin
    return lo, hi


# ---------------------------------------------------------------------------------------------------
# Device code
# ---------------------------------------------------------------------------------------------------

# Level sets, ported from ext/cusvna/examples/cuda/levelset.cu (same shapes, same animation). They are
# written once as templates on the evaluation type V (float or cusvna affine forms), the functions of
# affine forms are found by argument dependent lookup in namespace cusvna and their float counterparts
# are declared here. Shapes are given in local coordinates, they are negative inside and fit in the
# cylinder x^2+z^2 <= 1, |y| <= 1. The animation time t only enters through float coefficients.
SHAPES = r"""
namespace levelset {
    __device__ inline float sqr(float x) { return x*x; }
    __device__ inline float abs(float x) { return fabsf(x); }
    __device__ inline float min(float x, float y) { return fminf(x, y); }
    __device__ inline float max(float x, float y) { return fmaxf(x, y); }
    __device__ inline float sin(float x) { return sinf(x); }
    __device__ inline float cos(float x) { return cosf(x); }
    __device__ inline float sqrt(float x) { return sqrtf(x); }
    __device__ inline float cbrt(float x) { return cbrtf(x); }
    __device__ inline float positive_part(float x) { return fmaxf(x, 0.0f); }

    template <typename X, typename Y, typename Z>
    __device__ auto length(const X& x, const Y& y, const Z& z) {
        return sqrt(sqr(x) + sqr(y) + sqr(z));
    }

    /* a ball carved by a flowing gyroid lattice */
    template <typename V>
    __device__ auto gyroid(const V& x, const V& y, const V& z, const float t) {
        const float k = 6.0f;
        const auto X = k*x;
        const auto Y = k*y - 0.8f*t;
        const auto Z = k*z;
        const auto g = sin(X)*cos(Y) + sin(Y)*cos(Z) + sin(Z)*cos(X);
        const float thickness = 0.35f + 0.25f*sinf(0.7f*t);
        const auto lattice = (abs(g) - thickness) * (1.0f/k);
        const auto ball = length(x, y, z) - 0.95f;
        return max(lattice, ball);
    }

    /* five soft objects with compact polynomial falloff, w(d) = (1 - d^2/R^2)^2 for d < R */
    template <typename V>
    __device__ auto metaballs(const V& x, const V& y, const V& z, const float t) {
        auto blob = [&](const float a, const float b, const float c, const float phase, const float R) {
            const float cx = 0.3f * sinf(a*t + phase);
            const float cy = 0.3f * cosf(b*t + 2.0f*phase);
            const float cz = 0.3f * sinf(c*t + 3.0f*phase);
            const auto d2 = sqr(x - cx) + sqr(y - cy) + sqr(z - cz);
            return sqr(positive_part(1.0f - d2*(1.0f/(R*R))));
        };
        return 0.25f - (blob(1.1f, 0.7f, 0.9f, 0.0f, 0.44f) + blob(0.6f, 1.3f, 0.5f, 1.3f, 0.40f)
                     + blob(0.9f, 0.4f, 1.4f, 2.6f, 0.36f) + blob(1.7f, 0.9f, 0.3f, 3.9f, 0.32f)
                     + blob(0.3f, 1.6f, 1.1f, 5.2f, 0.34f));
    }

    /* a rounded box twisted around the y axis */
    template <typename V>
    __device__ auto twisted_box(const V& x, const V& y, const V& z, const float t) {
        const float twist = 2.2f*sinf(0.6f*t);
        const auto a = twist*y;
        const auto c = cos(a);
        const auto s = sin(a);
        const auto u = c*x - s*z;
        const auto w = s*x + c*z;
        const float r = 0.12f;
        const auto qx = abs(u) - 0.3f;
        const auto qy = abs(y) - 0.78f;
        const auto qz = abs(w) - 0.3f;
        const auto outside = length(positive_part(qx), positive_part(qy), positive_part(qz));
        const auto inside = min(max(qx, max(qy, qz)), 0.0f);
        return outside + inside - r;
    }

    /* Kummer quartic surface clipped by a ball, mu morphs it around its 16 nodes configuration (mu = 1.3) */
    template <typename V>
    __device__ auto kummer(const V& x0, const V& y0, const V& z0, const float t) {
        const float k = 2.4f;
        const auto x = k*x0;
        const auto y = k*z0;
        const auto z = k*y0;
        const float mu2 = sqr(1.3f + 0.2f*sinf(0.5f*t));
        const float lambda = (3.0f*mu2 - 1.0f) / (3.0f - mu2);
        const float s2 = 1.41421356f;
        const auto p = 1.0f - z - s2*x;
        const auto q = 1.0f - z + s2*x;
        const auto r = 1.0f + z + s2*y;
        const auto s = 1.0f + z - s2*y;
        const auto f = sqr(sqr(x) + sqr(y) + sqr(z) - mu2) - lambda*((p*q)*(r*s));
        const auto ball = sqr(x) + sqr(y) + sqr(z) - sqr(0.95f*k);
        return max(f, ball);
    }

    /* Barth sextic (65 nodes for w = 1) clipped by a ball, w breathes around 1 */
    template <typename V>
    __device__ auto barth(const V& x0, const V& y0, const V& z0, const float t) {
        const float k = 1.9f;
        const auto x = k*x0;
        const auto y = k*y0;
        const auto z = k*z0;
        const float phi = 1.61803399f;
        const float phi2 = phi*phi;
        const float w = 1.0f + 0.12f*sinf(0.8f*t);
        const auto xx = sqr(x);
        const auto yy = sqr(y);
        const auto zz = sqr(z);
        const auto f = 4.0f*((phi2*xx - yy)*(phi2*yy - zz))*(phi2*zz - xx)
                     - (1.0f + 2.0f*phi)*sqr(xx + yy + zz - w*w);
        const auto ball = xx + yy + zz - sqr(0.95f*k);
        return max(-f, ball);
    }

    /* Taubin heart surface (x^2 + 9/4 y^2 + z^2 - 1)^3 - z^3 (x^2 + 9/80 y^2) = 0, beating. The cube root
       of the equation has the same sign and a gradient that does not vanish on the surface where z = 0. */
    template <typename V>
    __device__ auto heart(const V& x0, const V& y0, const V& z0, const float t) {
        const float beat = 1.0f + 0.12f*powf(fabsf(sinf(2.5f*t)), 6.0f);
        const float k = 1.35f/beat;
        const auto x = k*x0;
        const auto y = k*z0;
        const auto z = k*y0 + 0.1f;
        return sqr(x) + 2.25f*sqr(y) + sqr(z) - 1.0f - z*cbrt(sqr(x) + 0.1125f*sqr(y));
    }

    /* the tetrahedral quartic r^4 + 8xyz - 10r^2 + c = 0, its handles open and close with c */
    template <typename V>
    __device__ auto tetrahedral(const V& x0, const V& y0, const V& z0, const float t) {
        const float k = 4.0f;
        const auto x = k*x0;
        const auto y = k*y0;
        const auto z = k*z0;
        const float c = 20.0f + 4.0f*sinf(0.6f*t);
        const auto r2 = sqr(x) + sqr(y) + sqr(z);
        return sqr(r2) + 8.0f*((x*y)*z) - 10.0f*r2 + c;
    }

    enum Shape : int {
        GYROID      = 1,
        METABALLS   = 2,
        TWISTED_BOX = 3,
        KUMMER      = 4,
        BARTH       = 5,
        HEART       = 6,
        TETRAHEDRAL = 7,
    };

    /* calls visitor(f) where f(x, y, z) is the level set function of the given shape at time t */
    template <typename Visitor>
    __device__ void visit_shape(const int shape, const float t, Visitor&& visitor) {
        switch (shape) {
            case GYROID:
                visitor([t](const auto& x, const auto& y, const auto& z) { return gyroid(x, y, z, t); });
                break;
            case METABALLS:
                visitor([t](const auto& x, const auto& y, const auto& z) { return metaballs(x, y, z, t); });
                break;
            case TWISTED_BOX:
                visitor([t](const auto& x, const auto& y, const auto& z) { return twisted_box(x, y, z, t); });
                break;
            case KUMMER:
                visitor([t](const auto& x, const auto& y, const auto& z) { return kummer(x, y, z, t); });
                break;
            case BARTH:
                visitor([t](const auto& x, const auto& y, const auto& z) { return barth(x, y, z, t); });
                break;
            case HEART:
                visitor([t](const auto& x, const auto& y, const auto& z) { return heart(x, y, z, t); });
                break;
            case TETRAHEDRAL:
                visitor([t](const auto& x, const auto& y, const auto& z) { return tetrahedral(x, y, z, t); });
                break;
            default:
                break;
        }
    }

    struct Object {
        float center[3];
        float scale;    /* the shape lives in [-scale, scale]^3 around center, in any rotation around y */
        float spin;     /* angular velocity around the y axis */
        float phase;    /* animation time offset */
        int   shape;
        float lo[3];    /* world bounding box */
        float hi[3];
        float albedo[3];
    };

    /* Rotation of an object around its vertical axis at time t, maps world coordinates to local
       coordinates. The mapping is affine so that it is exact for affine forms. */
    struct Frame {
        float cx, cy, cz, inv_scale, c, s;

        __device__ Frame(const Object& object, const float t)
            : cx(object.center[0]), cy(object.center[1]), cz(object.center[2]), inv_scale(1.0f/object.scale) {
            sincosf(object.spin * t, &s, &c);
        }

        /* evaluates the local function f at world coordinates (x, y, z) */
        template <typename V, typename F>
        __device__ auto eval(const F& f, const V& x, const V& y, const V& z) const {
            const auto dx = x - cx;
            const auto dy = y - cy;
            const auto dz = z - cz;
            return f(inv_scale*(c*dx - s*dz), inv_scale*dy, inv_scale*(s*dx + c*dz));
        }

        /* local gradient to world normal (inverse rotation) */
        __device__ void to_world(float& nx, float& ny, float& nz) const {
            const float x = c*nx + s*nz;
            const float z = -s*nx + c*nz;
            nx = x;
            nz = z;
        }
    };
} /* end of namespace levelset */
"""

# The scalar field, the normals and the colors. The extra kernel parameter t is the animation time.
FIELD = r"""
namespace levelset {
    /* index of the object whose bounding box contains (x, y, z), -1 if there is none (boxes are disjoint) */
    __device__ inline int find_object(const float x, const float y, const float z) {
        for (int i = 0; i < object_count; ++i) {
            const Object& o = objects[i];
            if ((x >= o.lo[0]) && (x <= o.hi[0]) && (y >= o.lo[1]) && (y <= o.hi[1])
                    && (z >= o.lo[2]) && (z <= o.hi[2]))
                return i;
        }
        return -1;
    }
}

namespace cupmb {
    /* union of the objects, positive outside of all the bounding boxes */
    template <typename T, typename I, I dim, I... N>
    __device__ __forceinline__ T scalar(
            const trove::array<T,dim>& p,
            const trove::array<I,dim>& i,
            const T& time) {
        using namespace levelset;
        const T x = trove::get<0>(p);
        const T y = trove::get<1>(p);
        const T z = trove::get<2>(p);
        const int k = find_object(x, y, z);
        T value = T(1);
        if (k >= 0) {
            const Object& o = objects[k];
            const float t = time + o.phase;
            const Frame frame(o, t);
            visit_shape(o.shape, t, [&](const auto& f) { value = frame.eval(f, x, y, z); });
        }
        return value;
    }

    /* normals point from negative to positive values (outward), computed by central differences in the
       local frame of the object (as in levelset.cu) and rotated back to world coordinates */
    template <typename T, typename I, I dim>
    __device__ __forceinline__ trove::array<T,dim> compute_normal(
            const trove::array<T,dim>& p,
            const T& time) {
        using namespace levelset;
        const T x = trove::get<0>(p);
        const T y = trove::get<1>(p);
        const T z = trove::get<2>(p);
        const int k = find_object(x, y, z);
        float nx = 0.0f, ny = 1.0f, nz = 0.0f;
        if (k >= 0) {
            const Object& o = objects[k];
            const float t = time + o.phase;
            const Frame frame(o, t);
            const float h = 2e-3f;
            visit_shape(o.shape, t, [&](const auto& f) {
                const auto g = [&](const float u, const float v, const float w) {
                    return frame.eval([&](const float a, const float b, const float c) {
                        return f(a + u, b + v, c + w); }, x, y, z);
                };
                nx = g(h, 0.0f, 0.0f) - g(-h, 0.0f, 0.0f);
                ny = g(0.0f, h, 0.0f) - g(0.0f, -h, 0.0f);
                nz = g(0.0f, 0.0f, h) - g(0.0f, 0.0f, -h);
            });
            frame.to_world(nx, ny, nz);
            const float n = rnorm3df(nx, ny, nz);
            nx *= n; ny *= n; nz *= n;
        }
        return trove::make_array<T>(nx, ny, nz);
    }

    template <typename T, typename I, I dim>
    __device__ __forceinline__ trove::array<T,dim> compute_color(
            const trove::array<T,dim>& p,
            const T& time) {
        using namespace levelset;
        const int k = find_object(trove::get<0>(p), trove::get<1>(p), trove::get<2>(p));
        if (k < 0)
            return trove::make_array<T>(0.8f, 0.8f, 0.8f);
        const Object& o = objects[k];
        return trove::make_array<T>(o.albedo[0], o.albedo[1], o.albedo[2]);
    }
}
"""

# Predicate definition (pdef): returns false only if the block certainly does not contain any zero of the
# scalar field. A block can only contain zeros inside the bounding box of an object, the shape of the
# object is then evaluated with affine arithmetic on the intersection of the block with the box. This is
# much tighter than a range analysis of the union: only the objects whose box meets the block are
# evaluated, on the part of the block that they cover. domain.a[k] and domain.b[k] are the lower and
# upper bounds of the block along axis k (0=x, 1=y, 2=z).
PDEF = r"""
#include <cusvna/af1.cuh>

namespace cupmb {
    template <typename T, typename I, I dim, typename... F>
    __device__ __forceinline__ bool may_contain_zeros(const BoundingBox<T,dim>& domain, const T& time) {
        using namespace levelset;
        using Interval = cusvna::Interval<float>;
        using AF = cusvna::AF1<float,3>;  /* 3 input noise symbols: x, y and z */

        for (int i = 0; i < object_count; ++i) {
            const Object& o = objects[i];
            const float xa = fmaxf(domain.a[0], o.lo[0]), xb = fminf(domain.b[0], o.hi[0]);
            const float ya = fmaxf(domain.a[1], o.lo[1]), yb = fminf(domain.b[1], o.hi[1]);
            const float za = fmaxf(domain.a[2], o.lo[2]), zb = fminf(domain.b[2], o.hi[2]);
            if ((xa > xb) || (ya > yb) || (za > zb))
                continue;

            const AF x = AF::template from_interval<0>(Interval(xa, xb));
            const AF y = AF::template from_interval<1>(Interval(ya, yb));
            const AF z = AF::template from_interval<2>(Interval(za, zb));

            const float t = time + o.phase;
            const Frame frame(o, t);
            bool zero = false;
            visit_shape(o.shape, t, [&](const auto& f) { zero = possibly_zero(frame.eval(f, x, y, z)); });
            if (zero)
                return true;
        }
        return false;
    }
}
"""

NORMAL = VertexProperty("normal", fname="compute_normal", dtype="T", shape=("dim",))
COLOR = VertexProperty("color", fname="compute_color", dtype="T", shape=("dim",))


def make_pmb(N, predicate, rigor, pow2=False):
    """
    Marching blocks on the bounding box of the scene, with cubic cells and N cells along x. With pow2, the
    number of cells on each axis is rounded up to a power of two and the domain is centered on the scene.
    """
    objects = make_scene()
    lo, hi = scene_domain(objects)
    L = hi - lo
    dx = L[0] / N
    Ns = np.ceil(L / dx).astype(np.int64)
    if pow2:
        Ns = 1 << np.ceil(np.log2(Ns)).astype(np.int64)
        lo = (lo + hi) / 2 - Ns * dx / 2
    return ParallelMarchingBlocks(
        3,
        SHAPES + make_object_table(objects) + FIELD,
        pdef=(PDEF if predicate else ""),
        x0=lo,
        L=Ns * dx,
        N=Ns,
        ftype=np.float32,
        itype=np.uint32,
        backend=Backend.CUPY,
        extra_params=[("t", "T")],
        vertex_properties=(NORMAL, COLOR),
        autotuning_rigor=rigor,
        enable_cuda_std_lib=True,
        # Disable FMA contraction so that the scalar field is evaluated identically by all kernel
        # variants: the triangle counts with and without predicate can then be compared exactly.
        extra_build_options=("--fmad=false",),
    )


# ---------------------------------------------------------------------------------------------------
# Headless benchmark
# ---------------------------------------------------------------------------------------------------
def benchmark(pmb, times):
    """Meshes the scene at the given animation times, returns the meshing times and triangle counts."""
    pmb(t=float(times[0]))  # build, autotune and preallocate buffers
    elapsed, triangles = [], []
    for t in times:
        pmb(t=float(t))  # (re)allocate the buffers if needed, the second call is timed
        (V, I, *_), dt = pmb(t=float(t), return_elapsed_time=True)
        elapsed.append(dt)
        triangles.append(0 if I is None else int(I.shape[0]))
    return np.asarray(elapsed), np.asarray(triangles)


def run_benchmark(args):
    times = np.linspace(0.0, 30.0, args.frames, endpoint=False) + 0.123
    rigor = AutotuningRigor[args.rigor]
    results = {}
    for predicate in ((False,) if args.no_predicate else (False, True)):
        pmb = make_pmb(args.N, predicate, rigor)
        elapsed, triangles = benchmark(pmb, times)
        name = "with" if predicate else "without"
        logger.info(
            f"N={tuple(map(int, pmb.N))}: median meshing time {time2str(float(np.median(elapsed)))} {name} predicate "
            f"({len(times)} frames, {int(triangles.min())} to {int(triangles.max())} triangles)."
        )
        results[predicate] = (elapsed, triangles)
        del pmb

    if args.no_predicate:
        return 0

    (t0, n0), (t1, n1) = results[False], results[True]
    logger.info(f"Speedup factor is {np.median(t0) / np.median(t1):2.1f} (ratio of the median meshing times).")
    mismatch = np.flatnonzero(n0 != n1)
    if mismatch.size > 0:
        for k in mismatch:
            logger.error(f"Triangle count mismatch at t={times[k]:.3f}: {n0[k]} without predicate vs {n1[k]} with.")
        return 1
    logger.info("Triangle counts are identical with and without predicate on all frames.")
    return 0


# ---------------------------------------------------------------------------------------------------
# OpenGL viewer
# ---------------------------------------------------------------------------------------------------

# Shading of levelset.cu without shadows: sky, sun light, hemispheric ambient, Blinn-Phong highlights,
# Fresnel reflection of the sky and distance fog.
GLSL_SHADING = """
uniform vec3 eye;

const vec3 sun_direction = normalize(vec3(-0.45, 0.75, -0.48));

vec3 sky(vec3 dir) {
    float h = max(dir.y, 0.0);
    vec3 color = mix(vec3(0.85, 0.72, 0.62), vec3(0.12, 0.25, 0.55), pow(h, 0.5));
    float sd = max(dot(dir, sun_direction), 0.0);
    color += vec3(1.0, 0.75, 0.45) * (0.25*pow(sd, 8.0) + 2.0*pow(sd, 512.0));
    if (dir.y < 0.0)
        color = mix(color, vec3(0.35, 0.3, 0.28), min(-4.0*dir.y, 1.0));
    return color;
}

vec3 to_srgb(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4)) - 0.055, step(0.0031308, c));
}

vec3 shade(vec3 p, vec3 normal, vec3 color, float shininess, bool reflective) {
    vec3 V = normalize(eye - p);
    vec3 H = normalize(sun_direction + V);
    float ndotl = dot(normal, sun_direction);
    vec3 sun_color = vec3(1.0, 0.92, 0.8) * 2.2;
    vec3 ambient = mix(vec3(0.25, 0.2, 0.18), vec3(0.3, 0.4, 0.6), 0.5 + 0.5*normal.y);
    float fresnel = 0.04 + 0.96*pow(1.0 - max(dot(normal, V), 0.0), 5.0);
    float specular = (ndotl > 0.0 ? 1.0 : 0.0) * pow(max(dot(normal, H), 0.0), shininess) * (shininess + 8.0) / 25.0;
    vec3 radiance = color * (ambient + sun_color * max(ndotl, 0.0)) + sun_color * specular * 0.25;
    if (reflective)
        radiance += fresnel * 0.5 * sky(reflect(-V, normal));
    float d = length(p - eye);
    float fog = 1.0 - exp(-0.0015 * d * d);
    return to_srgb(mix(radiance, sky(-V), fog));
}
"""

MESH_VS = """
#version 330
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
uniform mat4 view_projection;
out vec3 v_position;
out vec3 v_normal;
out vec3 v_color;
void main() {
    v_position = position;
    v_normal = normal;
    v_color = color;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""

MESH_FS = (
    """
#version 330
in vec3 v_position;
in vec3 v_normal;
in vec3 v_color;
out vec4 out_color;
"""
    + GLSL_SHADING
    + """
void main() {
    vec3 n = normalize(v_normal);
    vec3 color = v_color;
    /* two sided surfaces: clipped algebraic surfaces show their back faces, tint them */
    if (dot(n, eye - v_position) < 0.0) {
        n = -n;
        color = mix(color, vec3(1.0, 0.95, 0.85), 0.6);
    }
    out_color = vec4(shade(v_position, n, color, 64.0, true), 1.0);
}
"""
)

FLOOR_VS = """
#version 330
layout(location = 0) in vec3 position;
uniform mat4 view_projection;
out vec3 v_position;
void main() {
    v_position = position;
    gl_Position = view_projection * vec4(position, 1.0);
}
"""

FLOOR_FS = (
    """
#version 330
in vec3 v_position;
out vec4 out_color;
"""
    + GLSL_SHADING
    + """
void main() {
    float checker = float((int(floor(v_position.x)) + int(floor(v_position.z))) & 1);
    vec3 color = mix(vec3(0.55, 0.53, 0.50), vec3(0.75, 0.73, 0.70), checker);
    out_color = vec4(shade(v_position, vec3(0.0, 1.0, 0.0), color, 16.0, false), 1.0);
}
"""
)

SKY_VS = """
#version 330
layout(location = 0) in vec2 position;
out vec2 v_ndc;
void main() {
    v_ndc = position;
    gl_Position = vec4(position, 1.0, 1.0);
}
"""

SKY_FS = (
    """
#version 330
in vec2 v_ndc;
out vec4 out_color;
uniform mat4 inv_view_projection;
"""
    + GLSL_SHADING
    + """
void main() {
    vec4 p = inv_view_projection * vec4(v_ndc, 1.0, 1.0);
    out_color = vec4(to_srgb(sky(normalize(p.xyz/p.w - eye))), 1.0);
}
"""
)


def perspective(fov_y, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_y) / 2)
    return np.array(
        [
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    w = eye - target
    w /= np.linalg.norm(w)
    u = np.cross(up, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    M = np.eye(4, dtype=np.float32)
    M[0, :3], M[1, :3], M[2, :3] = u, v, w
    M[:3, 3] = -M[:3, :3] @ eye
    return M


class OrbitCamera:
    """Camera orbiting around a target point: left button drag to rotate, wheel to zoom."""

    def __init__(self, eye=(0.0, 6.0, -12.5), target=(0.0, -0.6, 0.0), fov_y=35.0):
        self.target = np.asarray(target, dtype=np.float32)
        offset = np.asarray(eye, dtype=np.float32) - self.target
        self.distance = float(np.linalg.norm(offset))
        self.pitch = float(np.arcsin(offset[1] / self.distance))
        self.yaw = float(np.arctan2(offset[0], -offset[2]))
        self.fov_y = fov_y

    @property
    def eye(self):
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        offset = self.distance * np.array([cp * np.sin(self.yaw), sp, -cp * np.cos(self.yaw)], dtype=np.float32)
        return self.target + offset

    def rotate(self, dx, dy):
        self.yaw += 0.005 * dx
        self.pitch = float(np.clip(self.pitch + 0.005 * dy, -1.5, 1.5))

    def zoom(self, steps):
        self.distance = float(np.clip(self.distance * 0.9**steps, 2.0, 60.0))

    def view_projection(self, aspect):
        return perspective(self.fov_y, aspect, 0.1, 200.0) @ look_at(self.eye, self.target)


# Bounds of the resolution of the viewer. The number of cells is a power of two on each axis (the domain is
# extended around the scene) and the '*' and '/' keys multiply or divide it by two: kernels depend on the
# grid size, they are compiled once for each resolution and then loaded from the kernel cache.
N_MIN, N_MAX = 64, 4096

CONTROLS = """\
SPACE   pause
+/-     animation speed (or UP/DOWN)
* /     resolution x2, /2 (or PAGE UP/DOWN)
W       wireframe
mouse   left drag to rotate, wheel to zoom
Q/ESC   quit"""


def run_viewer(args):
    import glfw
    from OpenGL import GL
    from OpenGL.GL import shaders

    try:
        import imgui
        from imgui.integrations.glfw import GlfwRenderer
    except ImportError:
        imgui = None  # no overlay, the controls are logged and the settings are shown in the window title

    predicate = not args.no_predicate
    # powers of two on each axis, so that each resolution of the viewer compiles its own kernels only once
    N = int(np.clip(1 << round(np.log2(args.N)), N_MIN, N_MAX))
    if N != args.N:
        logger.info(f"Viewer resolutions are powers of two, using N={N} instead of N={args.N}.")
    pmb = make_pmb(N, predicate, AutotuningRigor[args.rigor], pow2=True)

    if not glfw.init():
        logger.error("Could not initialize glfw.")
        return 1
    glfw.window_hint(glfw.SAMPLES, 4)
    title = f"cupmb real time level sets (predicate {'on' if predicate else 'off'})"
    window = glfw.create_window(1280, 720, title, None, None)
    if not window:
        glfw.terminate()
        logger.error("Could not create a glfw window.")
        return 1
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    if imgui is not None:
        imgui.create_context()
        overlay = GlfwRenderer(window, attach_callbacks=False)  # the callbacks of the viewer are set below
        imgui.get_style().window_border_size = 0.0

    def program(vs, fs):
        return shaders.compileProgram(
            shaders.compileShader(vs, GL.GL_VERTEX_SHADER), shaders.compileShader(fs, GL.GL_FRAGMENT_SHADER)
        )

    mesh_program = program(MESH_VS, MESH_FS)
    floor_program = program(FLOOR_VS, FLOOR_FS)
    sky_program = program(SKY_VS, SKY_FS)

    def static_vao(data, components):
        vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(vao)
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, components, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
        return vao

    s = 60.0
    floor_vao = static_vao(
        np.array([[-s, FLOOR_Y, -s], [s, FLOOR_Y, -s], [s, FLOOR_Y, s], [-s, FLOOR_Y, s]], dtype=np.float32), 3
    )
    sky_vao = static_vao(np.array([[-1, -1], [3, -1], [-1, 3]], dtype=np.float32), 2)

    # OpenGL buffers shared with cupmb: vertices, normals and colors (3 floats each) and triangles (3 uint32).
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    mesh_vao = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(mesh_vao)

    class SharedBuffer:
        def __init__(self, key, target, location=None):
            self.key, self.target, self.location = key, target, location
            self.gl_buffer = GL.glGenBuffers(1)
            self.cuda_buffer = None
            self.capacity = 0

        def resize(self, capacity):
            if self.cuda_buffer is not None:
                self.cuda_buffer.unregister()
            GL.glBindVertexArray(mesh_vao)
            GL.glBindBuffer(self.target, self.gl_buffer)
            GL.glBufferData(self.target, 12 * capacity, None, GL.GL_DYNAMIC_DRAW)
            if self.location is not None:
                GL.glEnableVertexAttribArray(self.location)
                GL.glVertexAttribPointer(self.location, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, ctypes.c_void_p(0))
            self.cuda_buffer = pmb.backend.CudaOpenGLMappedBuffer(self.gl_buffer, flags)
            self.capacity = capacity

    vertex_buffers = [
        SharedBuffer(BufferType.VERTICES, GL.GL_ARRAY_BUFFER, 0),
        SharedBuffer(NORMAL, GL.GL_ARRAY_BUFFER, 1),
        SharedBuffer(COLOR, GL.GL_ARRAY_BUFFER, 2),
    ]
    index_buffer = SharedBuffer(BufferType.SIMPLICES, GL.GL_ELEMENT_ARRAY_BUFFER)
    shared_buffers = vertex_buffers + [index_buffer]

    def mesh(t):
        """Meshes the scene at time t directly into the OpenGL buffers, grows them if they are too small."""
        for buf in shared_buffers:
            pmb.replace_buffer(buf.key, buf.cuda_buffer.map())
        try:
            (V, I, *_), elapsed = pmb(t=t, return_elapsed_time=True)
        finally:
            for buf in shared_buffers:
                buf.cuda_buffer.unmap()
        nv = 0 if V is None else int(V.shape[0])
        nt = 0 if I is None else int(I.shape[0])
        if (nv > vertex_buffers[0].capacity) or (nt > index_buffer.capacity):
            # cupmb wrote the mesh into its own (larger) buffers instead, grow the OpenGL ones and retry
            logger.info(f"Growing OpenGL buffers to {3 * nv // 2} vertices and {3 * nt // 2} triangles.")
            for buf in vertex_buffers:
                buf.resize(3 * nv // 2)
            index_buffer.resize(3 * nt // 2)
            return mesh(t)
        return nt, elapsed

    for buf in shared_buffers:
        buf.resize(1 << 20)

    camera = OrbitCamera()
    state = dict(paused=False, wireframe=False, drag=None, speed=1.0, N=N, requested_N=None)

    def change_speed(factor):
        state["speed"] = float(np.clip(state["speed"] * factor, 1.0 / 64, 64.0))

    def change_resolution(up):
        N = state["requested_N"] or state["N"]
        N = min(max(2 * N if up else N // 2, N_MIN), N_MAX)
        state["requested_N"] = None if N == state["N"] else N

    def rebuild(N):
        """Replaces the marching blocks instance by a new one with N cells along x (estimated block sizes)."""
        nonlocal pmb
        logger.info(f"Building the kernels for N={N}...")
        pmb = None  # releases the buffers of the previous instance before allocating the new ones
        cp.get_default_memory_pool().free_all_blocks()
        pmb = make_pmb(N, predicate, AutotuningRigor.ESTIMATE, pow2=True)
        state["N"] = N

    def on_key(window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            return
        if key in (glfw.KEY_UP, glfw.KEY_DOWN):  # repeated while held
            change_speed(1.25 if key == glfw.KEY_UP else 1 / 1.25)
        elif action != glfw.PRESS:
            return
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key in (glfw.KEY_PAGE_UP, glfw.KEY_PAGE_DOWN):
            change_resolution(key == glfw.KEY_PAGE_UP)

    def on_char(window, codepoint):
        # letters and symbols are handled as characters: glfw key codes are physical US positions (KEY_W is Z
        # on AZERTY, '+' and '*' are typed with different keys depending on the layout)
        c = chr(codepoint).lower()
        if c == "q":
            glfw.set_window_should_close(window, True)
        elif c == "w":
            state["wireframe"] = not state["wireframe"]
        elif c in "+=":
            change_speed(1.25)
        elif c == "-":
            change_speed(1 / 1.25)
        elif c == "*":
            change_resolution(True)
        elif c == "/":
            change_resolution(False)

    def on_mouse_button(window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            state["drag"] = glfw.get_cursor_pos(window) if action == glfw.PRESS else None

    def on_cursor(window, x, y):
        if state["drag"] is not None:
            x0, y0 = state["drag"]
            camera.rotate(x - x0, y - y0)
            state["drag"] = (x, y)

    def on_scroll(window, dx, dy):
        camera.zoom(dy)

    glfw.set_key_callback(window, on_key)
    glfw.set_char_callback(window, on_char)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)

    logger.info("Controls:\n" + CONTROLS)

    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glEnable(GL.GL_MULTISAMPLE)

    logger.info("Building the kernels and meshing the first frame...")
    mesh(0.0)

    def settings(requested_N):
        paused = " (paused)" if state["paused"] else ""
        lines = [
            f"N        : {' x '.join(map(str, map(int, pmb.N)))} cells",
            f"predicate: {'on' if predicate else 'off'}",
            f"speed    : x{state['speed']:.2f}{paused}",
        ]
        if requested_N is not None:
            lines.append(f"building the kernels for N={requested_N}...")
        return "\n".join(lines)

    def draw_overlay(text):
        overlay.process_inputs()
        imgui.new_frame()
        imgui.set_next_window_bg_alpha(0.4)
        imgui.set_next_window_position(10.0, 10.0)
        imgui.begin(
            "Overlay",
            None,
            imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_SCROLLBAR
            | imgui.WINDOW_NO_SAVED_SETTINGS
            | imgui.WINDOW_NO_INPUTS
            | imgui.WINDOW_ALWAYS_AUTO_RESIZE,
        )
        imgui.text(text)
        imgui.end()
        imgui.render()
        overlay.render(imgui.get_draw_data())

    anim_time = 0.0
    last = start = glfw.get_time()
    stats_start, stats_frames, stats_meshing = last, 0, 0.0
    stats_text = "fps      : -\nmeshing  : -\ntriangles: -"
    frames, total_meshing = 0, 0.0
    while not glfw.window_should_close(window):
        now = glfw.get_time()
        if not state["paused"]:
            anim_time += (now - last) * state["speed"]
        last = now

        ntriangles, elapsed = mesh(anim_time)
        stats_frames += 1
        stats_meshing += elapsed
        total_meshing += elapsed

        if now - stats_start >= 0.5:
            fps = stats_frames / (now - stats_start)
            meshing = stats_meshing / stats_frames
            stats_text = f"fps      : {fps:.0f}\nmeshing  : {1e3 * meshing:.1f} ms\ntriangles: {ntriangles / 1e6:.2f}M"
            glfw.set_window_title(
                window,
                f"{title} - N={state['N']} - {fps:.0f} fps - meshing {1e3 * meshing:.1f} ms - "
                f"{ntriangles / 1e6:.2f}M triangles - speed x{state['speed']:.2f}"
                + (" (paused)" if state["paused"] else ""),
            )
            stats_start, stats_frames, stats_meshing = now, 0, 0.0

        # a new resolution is applied after the frame announcing it has been displayed
        requested_N = state["requested_N"]

        width, height = glfw.get_framebuffer_size(window)
        if width == 0 or height == 0:
            glfw.poll_events()
            continue
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        eye = camera.eye
        VP = camera.view_projection(width / height)

        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glUseProgram(sky_program)
        GL.glUniformMatrix4fv(
            GL.glGetUniformLocation(sky_program, "inv_view_projection"), 1, GL.GL_TRUE, np.linalg.inv(VP)
        )
        GL.glUniform3fv(GL.glGetUniformLocation(sky_program, "eye"), 1, eye)
        GL.glBindVertexArray(sky_vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glDepthMask(GL.GL_TRUE)

        for prog in (floor_program, mesh_program):
            GL.glUseProgram(prog)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(prog, "view_projection"), 1, GL.GL_TRUE, VP)
            GL.glUniform3fv(GL.glGetUniformLocation(prog, "eye"), 1, eye)

        GL.glUseProgram(floor_program)
        GL.glBindVertexArray(floor_vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

        if state["wireframe"]:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glUseProgram(mesh_program)
        GL.glBindVertexArray(mesh_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 3 * ntriangles, GL.GL_UNSIGNED_INT, None)

        if imgui is not None:
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            draw_overlay(f"{stats_text}\n\n{settings(requested_N)}\n\n{CONTROLS}")
        elif requested_N is not None:
            glfw.set_window_title(window, f"{title} - building the kernels for N={requested_N}...")

        glfw.swap_buffers(window)
        glfw.poll_events()
        frames += 1

        if requested_N is not None:
            N, state["requested_N"] = state["requested_N"], None
            if N is not None:
                previous_N = state["N"]
                try:
                    rebuild(N)
                    mesh(anim_time)
                except cp.cuda.memory.OutOfMemoryError as e:
                    logger.error(f"Not enough device memory for N={N} ({e}), going back to N={previous_N}.")
                    rebuild(previous_N)
            # the build time is neither animated nor accounted in the statistics
            last = stats_start = glfw.get_time()
            stats_frames, stats_meshing = 0, 0.0

        if args.frames and frames >= args.frames:
            break

    if frames > 0:
        fps = frames / (glfw.get_time() - start)
        logger.info(f"{frames} frames, {fps:.1f} fps on average, mean meshing time {time2str(total_meshing / frames)}.")
    if imgui is not None:
        overlay.shutdown()
    for buf in shared_buffers:
        buf.cuda_buffer.unregister()
    glfw.terminate()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real time meshing of animated level sets with a block predicate.")
    parser.add_argument(
        "-N",
        type=int,
        default=2560,
        help="number of grid cells along x (the longest axis), rounded to a power of two by the viewer",
    )
    parser.add_argument("--no-predicate", action="store_true", help="disable the block predicate")
    parser.add_argument(
        "--rigor",
        default="ESTIMATE",
        choices=[r.name for r in AutotuningRigor],
        help="autotuning rigor",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="number of frames (default: 32 with --no-display, unlimited otherwise)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="headless benchmark: meshes the scene at different times with and without predicate",
    )
    args = parser.parse_args()

    if args.no_display:
        args.frames = args.frames or 32
        sys.exit(run_benchmark(args))
    else:
        sys.exit(run_viewer(args))
