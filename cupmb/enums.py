#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#
import enum


class Enum(enum.Enum):
    def __str__(self) -> str:
        return self.name


class Backend(enum.Enum):
    CUPY = 0
    PYCUDA = 1


class BufferType(Enum):
    COUNTERS = 0
    SCALAR_FIELD = 1
    BLOCK_MASK = 2
    BLOCK_INDICES_PING = 3
    BLOCK_INDICES_PONG = 4
    VERTICES = 5
    SIMPLICES = 6


class AllocationPolicy(Enum):
    LOCK_BUFFER = 0
    NO_EXTRA_ALLOCATION = 1
    ALLOCATE_EXACT_SIZE = 2
    ALLOCATE_UPPER_POW2 = 3


class AutotuningRigor(enum.IntEnum):
    """Autotuning rigor, ordered from the cheapest to the most expensive (comparable)."""

    def __str__(self) -> str:
        return self.name

    ESTIMATE = 0
    MEASURE = 1
    PATIENT = 2
    EXHAUSTIVE = 3
