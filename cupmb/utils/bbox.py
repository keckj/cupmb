#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
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
        return open3d.geometry.AxisAlignedBoundingBox.create_from_points(
            open3d.utility.Vector3dVector(np.vstack([self.xmin, self.xmax])),
        )
