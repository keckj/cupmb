#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
import numpy as np


class BoundingBox:
    def __init__(self, xmin, xmax):
        """Initialize a bounding box from a min and max point."""
        assert isinstance(xmin, np.ndarray), type(xmin)
        assert isinstance(xmax, np.ndarray), type(xmin)
        assert xmin.ndim == 1, xmin.ndim
        assert xmax.ndim == 1, xmax.ndim
        assert xmin.size == xmax.size, (xmin.size, xmax.size)
        assert xmin.dtype == xmax.dtype, (xmin.dtype, xmax.dtype)
        assert (xmax >= xmin).all(), xmax - xmin
        self._xmin = xmin
        self._xmax = xmax

    def __str__(self):
        return f"BoundingBox(xmin={self.xmin}, xmax={self.xmax}, L={self.length}, dtype={self.dtype})"

    def _get_xmin(self):
        """Get the starting point of this bounding box."""
        return self._xmin

    def _set_xmin(self, value):
        """Set the starting point of this bounding box."""
        self._xmin[...] = value

    xmin = property(_get_xmin, _set_xmin)

    def _get_xmax(self):
        """Get the end point of this bounding box."""
        return self._xmax

    def _set_xmax(self, value):
        """Set the end point of this bounding box."""
        self._xmax[...] = value

    xmax = property(_get_xmax, _set_xmax)

    def _astype(self, dtype):
        """Cast this bounding box to another floating point type, in place."""
        self._xmin = self._xmin.astype(dtype)
        self._xmax = self._xmax.astype(dtype)
        return self

    @property
    def dim(self):
        """Get the dimension of this bounding box."""
        return self._xmin.size

    @property
    def dtype(self):
        """Get the datatype of this bounding box."""
        return self._xmin.dtype

    @property
    def length(self):
        """Get the physical size of this bounding box."""
        return self._xmax - self._xmin

    @property
    def as_open3d(self):
        """Returns an AxisAlignedBoundingBox representing this bounding box in open3d."""
        import open3d

        assert self.dim == 3, self.dim
        bbox = open3d.geometry.AxisAlignedBoundingBox.create_from_points(
            open3d.utility.Vector3dVector(np.vstack([self.xmin, self.xmax]).astype(np.float64)),
        )
        bbox.color = (1.0, 0.0, 0.0)
        return bbox
