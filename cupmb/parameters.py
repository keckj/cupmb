#
# Copyright (C) 2021 Jean-Baptiste Keck - All Rights Reserved
#
# Project CUPMB and package cupmb are proprietary and confidential
# and can not be copied, reproduced, modified and/or distributed without the express permission
# of all copyright holders (Jean-Baptiste Keck).
#
# Written by Jean-Baptiste Keck <jbkeck@hotmail.com>, August 2021.
#


class ExtraParameter:
    def __init__(self, ctype, update=None):
        if update is not None:
            assert callable(update), type(update)
        self.ctype = ctype
        self.update = update

    def is_callable(self):
        return callable(self.update)

    def __call__(self, *args, **kwds):
        assert self.is_callable, "ExtraParameter is not callable."
        return self.update(*args, **kwds)
