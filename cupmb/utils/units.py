#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#

from typing import Union, Optional

import numpy as np

Number = Union[int, float, np.number]


def binary_unit2str(b: Number, unit: str, fmt: str = "{:.1f}") -> str:
    """Convert any unit to a pretty string in binary base."""
    try:
        bb = int(b)
        prefix = ["", "ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
        i = 0
        while bb >= 1024:
            bb >>= 10
            i += 1
        fmt = fmt if prefix[i] else "{:d}"
        return (fmt + "{}{}").format(b / (1024 ** i), prefix[i], unit)
    except:
        return (fmt + "{}").format(b, unit)


def decimal_unit2str(b: Number, unit: str, fmt: str = "{:.1f}") -> str:
    """Convert any unit to a pretty string in decimal base."""
    try:
        if b == 0:
            fmt = "{:d}"
            return fmt.format(0) + unit
        elif abs(b) >= 1:  # type: ignore
            bb = int(abs(b))  # type: ignore
            prefix = ["", "k", "M", "G", "T", "P", "E", "Z", "Y"]
            i = 0
            while bb >= 1000:
                bb /= 1000
                i += 1
            fmt = fmt if prefix[i] else "{:d}"
            return (fmt + "{}{}").format(b / (10.0 ** (3 * i)), prefix[i], unit)
        else:
            prefix = ["", "m", "u", "n", "p", "f", "a", "z", "y"]
            i = 0
            bb = b
            while abs(bb) < 1:
                bb *= 1000
                i += 1
            fmt = fmt if prefix[i] else "{:d}"
            return (fmt + "{}{}").format(b * (10.0 ** (3 * i)), prefix[i], unit)
    except:
        return (fmt + "{}").format(b, unit)


def unit2str(
    b: Optional[Number], unit: str, decimal: bool, fmt: str = "{:2.1f}"
) -> str:
    """Convert any unit to a pretty string in binary or decimal base."""

    if not isinstance(b, (int, float, np.number)):
        return str(b)

    if decimal:
        return decimal_unit2str(b, unit, fmt)
    else:
        return binary_unit2str(b, unit, fmt)


def bytes2str(b: Optional[Number], decimal: bool = False, fmt: str = "{:2.1f}") -> str:
    """Convert input bytes to a pretty string."""
    return unit2str(b, "B", decimal, fmt)


def bdw2str(bdw: Optional[Number], decimal: bool = True, fmt: str = "{:2.1f}") -> str:
    """Convert an input memory bandwidth in bytes per second to a pretty string."""
    return unit2str(bdw, "B/s", decimal, fmt)


def time2str(
    t: Optional[Number], on_zero: Optional[str] = None, on_none: Optional[str] = None
) -> str:
    """Convert an input time in seconds to a pretty string."""
    if not isinstance(t, (float, int, np.dtype)):
        return str(t)
    if t is None:
        return f"{-1:5d}" if (on_none is None) else on_none
    elif t < 1e-9:
        return f"{0:5d}" if (on_zero is None) else on_zero
    elif t < 1e-6:
        return f"{t * 1e9:>5.1f}ns"
    elif t < 1e-3:
        return f"{t * 1e6:>5.1f}us"
    elif t < 1e1:
        return f"{t * 1e3:>5.1f}ms"
    else:
        return f"{t:>6.1f}s"
