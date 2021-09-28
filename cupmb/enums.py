#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
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
    VERTICES = 2
    SIMPLICES = 3
    BLOCK_INDICES = 4
    BLOCK_MASK = 5


class AllocationPolicy(Enum):
    LOCK_BUFFER = 0
    NO_EXTRA_ALLOCATION = 1
    ALLOCATE_EXACT_SIZE = 2
    ALLOCATE_UPPER_POW2 = 3


class AutotuningRigor(Enum):
    ESTIMATE = 0
    MEASURE = 1
    PATIENT = 2
    EXHAUSTIVE = 3
