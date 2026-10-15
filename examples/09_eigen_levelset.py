#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#

# Real time meshing of level sets written with Eigen matrices.
#
# The scene is the one of the Eigen example of cusvna (examples/06_eigen_levelset.py and
# examples/cuda/eigen_levelset.cu in https://github.com/keckj/cusvna): a Menger sponge between a
# spectrahedron and a trefoil knot. The shapes are written with Eigen vectors and matrices (rotations,
# quaternions, Householder reflections, determinants, quadratic forms) of floats for the scalar field,
# of cusvna affine forms cusvna::AF1<float,3> for the block predicate and of autodiff numbers for the
# normals. Eigen is compiled by NVRTC with the eigen-nvrtc git submodule (ext/eigen-nvrtc).
#
# Only AF1 forms can be Eigen scalars: the type of an AA form records its noise symbols and changes with
# each nonlinear operation, while the coefficients of an Eigen matrix share the same type.
#
# Same viewer, benchmark and options as 08_levelset.py.
#
import os
import sys
import enum
import importlib

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
levelset = importlib.import_module("08_levelset")

EIGEN_INCLUDE_DIR = os.path.realpath(os.path.join(script_dir, "..", "ext", "eigen-nvrtc"))


class Shape(enum.IntEnum):
    """Must match the Shape enum of the device code below."""

    MENGER = 1
    SPECTRAHEDRON = 2
    TREFOIL = 3


ALBEDO = {
    Shape.MENGER: (0.95, 0.62, 0.20),
    Shape.SPECTRAHEDRON: (0.10, 0.55, 0.95),
    Shape.TREFOIL: (0.90, 0.15, 0.50),
}


def make_scene():
    """A Menger sponge between a spectrahedron and a trefoil knot: (center, scale, spin, shape, phase)."""
    floor_y = levelset.FLOOR_Y
    return [
        ((0.0, floor_y + 0.97, 0.0), 1.4, 0.2, Shape.MENGER, 0.0),
        ((-3.2, floor_y + 0.67, 0.6), 0.95, -0.4, Shape.SPECTRAHEDRON, 0.0),
        ((3.2, floor_y + 0.9, 0.6), 0.95, 0.3, Shape.TREFOIL, 0.0),
    ]


# Level sets ported from examples/cuda/eigen_levelset.cu of cusvna (same shapes, same animation). They are
# function objects whose call operator is a template on the scalar type V: float, affine form or autodiff
# number. They work on Eigen vectors and matrices of V, mixed with matrices of floats (rotations, folding
# planes) computed by the constructor.
SHAPES = r"""
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <autodiff/forward/real/eigen.hpp>
#include <cusvna/af1.cuh>

/* Eigen matrices of AF1 forms, mixed with matrices of floats in products and coefficient-wise operations.
   autodiff/forward/real/eigen.hpp does the same for autodiff::Real. */
namespace Eigen {
    template <typename T, int N>
    struct NumTraits<cusvna::AF1<T,N>> : NumTraits<T> {
        using Real = cusvna::AF1<T,N>;
        using NonInteger = cusvna::AF1<T,N>;
        using Nested = cusvna::AF1<T,N>;
        using Literal = T;
        enum {
            IsComplex = 0,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 4,
            MulCost = 8,
        };
    };

    template <typename T, int N, typename BinaryOp>
    struct ScalarBinaryOpTraits<cusvna::AF1<T,N>, T, BinaryOp> { using ReturnType = cusvna::AF1<T,N>; };

    template <typename T, int N, typename BinaryOp>
    struct ScalarBinaryOpTraits<T, cusvna::AF1<T,N>, BinaryOp> { using ReturnType = cusvna::AF1<T,N>; };
} /* end of namespace Eigen */

namespace levelset {
    template <typename V>
    using Vector3 = Eigen::Matrix<V, 3, 1>;

    template <typename V>
    __device__ V max_norm(const Vector3<V>& p) {
        return max(max(abs(p.x()), abs(p.y())), abs(p.z()));
    }

    /* Menger sponge as a kaleidoscopic iterated function system: each iteration folds space with
       p = |p| and three Householder reflections that sort the coordinates, rotates it and scales it
       around a corner of the cube, which maps the 20 subcubes of the sponge to the cube. The rotation
       slowly moves away from the identity (the classical sponge) and back. */
    struct Menger {
        static constexpr int iterations = 3;
        static constexpr float size = 0.68f;  /* half side of the cube, its corners fit in the unit cylinder */
        Eigen::Matrix3f R;

        __device__ explicit Menger(const float t) {
            const Eigen::Quaternionf q(Eigen::AngleAxisf(0.55f, Eigen::Vector3f(1.0f, 2.0f, 3.0f).normalized()));
            R = Eigen::Quaternionf::Identity().slerp(0.5f - 0.5f*cosf(0.4f*t), q).toRotationMatrix();
        }

        template <typename V>
        __device__ V operator()(const V& x, const V& y, const V& z) const {
            const float h = 0.70710678f;
            const Eigen::Vector3f folds[3] = {{h, -h, 0.0f}, {h, 0.0f, -h}, {0.0f, h, -h}};

            Vector3<V> p(x, y, z);
            p *= 1.0f/size;
            const V cube = max_norm(p) - 1.0f;
            for (int i = 0; i < iterations; ++i) {
                p = p.cwiseAbs();
                for (const Eigen::Vector3f& n : folds)
                    p -= (2.0f*min(n.dot(p), 0.0f)) * n;
                p = R*p;
                p = 3.0f*p - Eigen::Vector3f(2.0f, 2.0f, 0.0f);
                p.z() = 1.0f - abs(p.z() - 1.0f);
            }
            const float scale = 1.0f / 27.0f;  /* 3^-iterations */
            return size * max((max_norm(p) - 1.0f) * scale, cube);
        }
    };

    /* A spectrahedron is the set where a symmetric matrix depending linearly on the coordinates is positive
       definite. For A = [[1, x, y, z], [x, 1, l*z, l*y], [y, l*z, 1, l*x], [z, l*y, l*x, 1]] it is the unit
       ball for l = 0 and the tetrahedron whose faces are the eigenvalues 1 +/- x +/- y +/- z of A for
       l = +/-1, l oscillates between them. A is positive definite if and only if its leading principal
       minors are positive (Sylvester's criterion), the smallest one is the level set function. */
    struct Spectrahedron {
        float lambda;

        __device__ explicit Spectrahedron(const float t) : lambda(sinf(0.5f*t)) {}

        template <typename V>
        __device__ V operator()(const V& x0, const V& y0, const V& z0) const {
            const float k = 1.45f;  /* the vertices (+/-1, +/-1, +/-1) of the tetrahedra fit in the unit cylinder */
            const V x = k*x0;
            const V y = k*y0;
            const V z = k*z0;
            const V one(1.0f);
            const V lx = lambda*x;
            const V ly = lambda*y;
            const V lz = lambda*z;
            Eigen::Matrix<V, 4, 4> A;
            A << one,   x,   y,   z,
                   x, one,  lz,  ly,
                   y,  lz, one,  lx,
                   z,  ly,  lx, one;
            const V m2 = A.template topLeftCorner<2, 2>().determinant();
            const V m3 = A.template topLeftCorner<3, 3>().determinant();
            const V m4 = A.determinant();
            return -min(min(m2, m3), m4);
        }
    };

    /* Ellipsoidal beads sliding along a trefoil knot, with their main axis along the knot and twisting
       around it. Each bead is the quadratic form |D R^T (p - c)|^2 < 1 where the rotation R is given by a
       quaternion and D is the diagonal of the inverse radii, the union is the minimum of the forms.
       (Quaternion::FromTwoVectors would be shorter, but it falls back to a JacobiSVD that calls malloc
       for opposite vectors.) */
    struct Trefoil {
        static constexpr int beads = 24;
        float t;

        __device__ explicit Trefoil(const float t) : t(t) {}

        template <typename V>
        __device__ V operator()(const V& x, const V& y, const V& z) const {
            const float k = 0.27f;
            const Eigen::Vector3f inv_radii(1.0f/0.2f, 1.0f/0.12f, 1.0f/0.06f);
            const Vector3<V> p(x, y, z);
            V d;
            for (int i = 0; i < beads; ++i) {
                const float s = (6.28318531f*i + 0.6f*t) / beads;
                float s1, c1, s2, c2, s3, c3;
                sincosf(s, &s1, &c1);
                sincosf(2.0f*s, &s2, &c2);
                sincosf(3.0f*s, &s3, &c3);
                const Eigen::Vector3f c = k*Eigen::Vector3f(s1 + 2.0f*s2, c1 - 2.0f*c2, -s3);
                /* the tangent is never vertical: the norm of its (x, y) part |exp(-is) + 4 exp(2is)| is at least 3 */
                const Eigen::Vector3f tangent = Eigen::Vector3f(c1 + 4.0f*c2, -s1 + 4.0f*s2, -3.0f*c3).normalized();
                const Eigen::Vector3f normal = tangent.cross(Eigen::Vector3f::UnitZ()).normalized();
                Eigen::Matrix3f frame;
                frame << tangent, normal, tangent.cross(normal);
                const Eigen::Quaternionf q = Eigen::Quaternionf(frame)
                                           * Eigen::AngleAxisf(3.0f*s + 1.5f*t, Eigen::Vector3f::UnitX());
                const Eigen::Matrix3f Rt = q.toRotationMatrix().transpose();
                const V e = (Rt*(p - c)).cwiseProduct(inv_radii).squaredNorm() - 1.0f;
                d = (i == 0 ? e : V(min(d, e)));
            }
            return d;
        }
    };

    enum Shape : int {
        MENGER        = 1,
        SPECTRAHEDRON = 2,
        TREFOIL       = 3,
    };

    /* calls visitor(f) where f(x, y, z) is the level set function of the given shape at time t */
    template <typename Visitor>
    __device__ void visit_shape(const int shape, const float t, Visitor&& visitor) {
        switch (shape) {
            case MENGER:
                visitor(Menger(t));
                break;
            case SPECTRAHEDRON:
                visitor(Spectrahedron(t));
                break;
            case TREFOIL:
                visitor(Trefoil(t));
                break;
            default:
                break;
        }
    }
} /* end of namespace levelset */
"""

# Eigen leaves many functions without __host__ __device__ annotations, NVRTC compiles them as device
# functions with -default-device.
SCENE = levelset.Scene(
    "cupmb Eigen level sets",
    SHAPES,
    make_scene(),
    ALBEDO,
    build_options=("-default-device", f"-I{EIGEN_INCLUDE_DIR}"),
    eye=(0.0, 3.2, -8.5),
    target=(0.0, -0.7, 0.3),
    benchmark_N=2560,
)

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(EIGEN_INCLUDE_DIR, "Eigen")):
        levelset.logger.error("Cannot find the eigen-nvrtc submodule, run git submodule update --init --recursive.")
        sys.exit(1)
    sys.exit(levelset.main(SCENE, "Real time meshing of level sets written with Eigen matrices."))
