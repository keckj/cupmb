#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package pmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#
from collections.abc import Iterable

import numpy as np


def get_dtype(x, signed=None):
    """
    Extract a numpy data type (a subclass of np.generic) from any compatible
    type of instance such as np.dtype, np.ndarray, int, float, complex.
    If input is None, returns None.
    """
    ret = None
    if isinstance(x, np.dtype):
        ret = x.type
    elif hasattr(x, "dtype"):
        if callable(x.dtype):
            ret = x.dtype()
        elif x.dtype.__class__.__name__ == "getset_descriptor":
            # dtype.type has a dtype field...
            ret = x
        else:
            ret = x.dtype
    elif isinstance(x, int) or (isinstance(x, type) and issubclass(x, int)):
        assert signed is not None
        if signed:
            ret = np.int64
        else:
            ret = np.uint64
    elif isinstance(x, float) or (isinstance(x, type) and issubclass(x, float)):
        ret = np.float64
    elif isinstance(x, complex) or (isinstance(x, type) and issubclass(x, complex)):
        ret = np.complex128
    elif x is None:
        ret = None
    else:
        msg = "Unknown type in get_dtype (got '{}' of type '{}')."
        msg = msg.format(getattr(x, "__name__", str(x)), type(x).__name__)
        raise TypeError(msg)
    return ret


def to_tuple(arg, cast=None):
    """
    Convert any of the following types (scalar, tuple, list, set, frozenset,
    np.ndarray, Iterable) to a tuple.
    """
    cast = first_not_None(cast, lambda x: x)
    if isinstance(arg, (str, dict, type)):
        return (arg,)
    elif hasattr(arg, "__tuple__"):
        return tuple(cast(x) for x in arg.__tuple__())
    elif isinstance(arg, np.ndarray):
        try:
            return tuple(cast(x) for x in arg.tolist())
        except:
            return (cast(arg),)
    elif isinstance(arg, (tuple, list, frozenset, set, Iterable)):
        return tuple(cast(x) for x in arg)
    else:
        return (cast(arg),)


def first_not_None(a0, *args):
    """
    Returns the first argument that is not None.
    If only one argument is given it is considered as an iterable,
    else iterates on (a0,)+args.
    If all input arguments are None, returns None.
    """
    if len(args) == 0:
        try:
            return next((a for a in a0 if (a is not None)), None)
        except TypeError:  # a0 is not iterable
            return a0
    else:
        return next((a for a in (a0,) + args if (a is not None)), None)
