# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Update rule functions."""
from .base import GenericUpdateRule
from .hebian import HebbianDense

__all__ = [
    'GenericUpdateRule', 'HebbianDense'
]