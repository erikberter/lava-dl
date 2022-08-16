# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch

from lava.lib.dl.slayer.utils.update_rule.stdp import LinearSTDPDense
from lava.lib.dl import slayer, netx
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.proc.conv import utils
from lava.proc import io

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

neuron_param = {'threshold': 0.5,
                'current_decay': 0.5,
                'voltage_decay': 0.5}

updatable_neuron_param = {
    'threshold': 0.5,
    'current_decay': 0.5,
    'voltage_decay': 0.5,
    'persistent_state': True}


class TestLinearSTDPDense(unittest.TestCase):
    """Test LinearSTDPDense blocks"""

    def test_linear_stdp_dense_creation(self):
        """Test if the LinearSTDP class can be correctly launched."""
        try:
            update_rule = LinearSTDPDense(1, 1, 1)
        except Exception:
            self.fail("LinearSTDPDense creation failed")

    def test_linear_stdp_dense_updates_correctly(self):
        """Test if the STDP dynamics are correctly implemented."""
        update_rule = LinearSTDPDense(1, 1, 1)

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]]])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.equal(weight, torch.FloatTensor([[0.9264]]))
