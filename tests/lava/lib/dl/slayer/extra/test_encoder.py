# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch

from lava.lib.dl import slayer, netx
from lava.magma.core.run_conditions import RunSteps
from lava.proc.conv import utils
from lava.proc import io

from lava.lib.dl.slayer.extra.encoder import GenericEncoder
from lava.lib.dl.slayer.extra.encoder import RateEncoder

verbose = True if (('-v' in sys.argv) or ('--verbose' in sys.argv)) else False
# Enabling torch sometimes causes multiprocessing error, especially in unittests
utils.TORCH_IS_AVAILABLE = False

# seed = np.random.randint(1000)
seed = 196
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if verbose:
    print(f'{seed=}')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    if verbose:
        print('CUDA is not available in the system. '
              'Testing for CPU version only.')
    device = torch.device('cpu')

tempdir = os.path.dirname(__file__) + '/temp'
os.makedirs(tempdir, exist_ok=True)


class TestGenericEncoder(unittest.TestCase):
    """Test Generic Encoder"""

    def test_generic_encoder_call_works(self):
        class CustomEncoder(GenericEncoder):
            def encode(self, input, num_steps=1):
                return input

        encoder = CustomEncoder()

        try:
            encoder(torch.zeros((3, 3, 3)))
        except Exception:
            self.fail("__call__ not working on generic encoder")


class TestRateEncoder(unittest.TestCase):
    """Test Rate Encoder"""

    def test_rate_encoder_creation(self):
        try:
            encoder = RateEncoder()
        except Exception:
            self.fail("Cannot initialize RateEncoder")

    def test_rate_encoder_results(self):
        encoder = RateEncoder()
        res = encoder(0.5 * torch.ones((4, 4, 4)), 3)
        assert 0.4 < res.mean() < 0.6
