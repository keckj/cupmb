#
# Copyright (c) 2021-2026 Jean-Baptiste Keck <jbkeck@hotmail.com>
# SPDX-License-Identifier: MIT
#


class ExtraParameter:
    def __init__(self, ctype, update=None):
        if update is not None:
            assert callable(update), type(update)
        self.ctype = ctype
        self.update = update

    def __eq__(self, other):
        if not isinstance(other, ExtraParameter):
            return NotImplemented
        return (self.ctype == other.ctype) and (self.update == other.update)

    def __hash__(self):
        return hash((self.ctype, self.update))

    def __repr__(self):
        return f"ExtraParameter(ctype={self.ctype!r}, update={self.update!r})"

    @property
    def is_callable(self):
        return callable(self.update)

    def __call__(self, *args, **kwds):
        assert self.is_callable, "ExtraParameter is not callable."
        return self.update(*args, **kwds)
