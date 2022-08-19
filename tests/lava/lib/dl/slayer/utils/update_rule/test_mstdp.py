# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch

from lava.lib.dl.slayer.utils.update_rule.mstdp import LinearMSTDPDense
from lava.lib.dl.slayer.utils.update_rule.mstdp import LinearMSTDPETDense

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


class TestLinearMSTDPDense(unittest.TestCase):
    """Test LinearMSTDPDense blocks"""

    def test_linear_mstdp_dense_creation(self):
        """Test if the LinearMSTDP class can be correctly launched."""
        try:
            update_rule = LinearMSTDPDense(1, 1, 1)
        except Exception:
            self.fail("LinearMSTDPDense creation failed")

    def test_linear_mstdp_dense_updates_correctly(self):
        """Test if the MSTDP dynamics are correctly implemented."""
        update_rule = LinearMSTDPDense(1, 1, 1, tau=0.5)

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]]])
        reward = torch.FloatTensor([-0.1, -0.1, 1, 0.5, 0.3, 0.2])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(
                weight,
                pre[:, :, t],
                post[:, :, t],
                reward=reward[t])

        assert torch.equal(weight, torch.FloatTensor([[1]]))

    def test_linear_mstdp_works_on_batch(self):
        """Test if the MSTDP update rule works on batch."""
        update_rule = LinearMSTDPDense(1, 1, 1, tau=0.5)

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]], [[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]], [[1, 0, 0, 1, 0, 0]]])
        reward = torch.FloatTensor([-0.1, -0.1, 1, 0.5, 0.3, 0.2])

        weight = torch.Tensor([[1]])

        try:
            for t in range(pre.shape[-1]):
                weight = update_rule.update(
                    weight,
                    pre[:, :, t],
                    post[:, :, t],
                    reward=reward[t])
        except Exception:
            self.fail("MSTDP Update rule is not working on batch")


class TestLinearMSTDPETDense(unittest.TestCase):
    """Test LinearMSTDPETDense blocks"""

    def test_linear_mstdpet_dense_creation(self):
        """Test if the LinearMSTDPET class can be correctly launched."""
        try:
            update_rule = LinearMSTDPETDense(1, 1, 1)
        except Exception:
            self.fail("LinearMSTDPETDense creation failed")

    def test_linear_mstdpet_dense_updates_correctly(self):
        """Test if the MSTDPET dynamics are correctly implemented."""
        update_rule = LinearMSTDPETDense(1, 1, 1, tau=0.5)

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]]])
        reward = torch.FloatTensor([-0.1, -0.1, 1, 0.5, 0.3, 0.2])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(
                weight,
                pre[:, :, t],
                post[:, :, t],
                reward=reward[t])

        assert torch.isclose(
            weight, torch.FloatTensor([[0.9998]]),
            rtol=1e-03,
            atol=1e-03)

    def test_linear_mstdpet_works_on_batch(self):
        """Test if the MSTDPET update rule works on batch."""
        update_rule = LinearMSTDPETDense(1, 1, 1, tau=0.5)

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]], [[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]], [[1, 0, 0, 1, 0, 0]]])
        reward = torch.FloatTensor([-0.1, -0.1, 1, 0.5, 0.3, 0.2])

        weight = torch.Tensor([[1]])

        try:
            for t in range(pre.shape[-1]):
                weight = update_rule.update(
                    weight,
                    pre[:, :, t],
                    post[:, :, t],
                    reward=reward[t])
        except Exception:
            self.fail("MSTDPET Update rule is not working on batch")
