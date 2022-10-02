# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch


from lava.lib.dl.slayer.utils.update_rule.homeostasis import Homeostasis

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


class TestHomeostasisDenseFunctional(unittest.TestCase):
    """Test HomeostasisDense blocks"""

    def test_homeostasis_dense_functional_creation(self):
        """Test if the HomeostasisDense class can be correctly launched."""
        try:
            update_rule = Homeostasis(1, 1, 1, tau=0.5)
        except Exception:
            self.fail("HomeostasisDense creation failed")

    def test_homeostasis_dense_functional_updates_on_zero_rate(self):
        """Test if the HomeostasisDense dynamics are correctly implemented."""
        update_rule = Homeostasis(1, 1, 1, tau=0.5, early_start=True)
        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 0, 0, 0, 0]]])
        reward = torch.FloatTensor([0, 0, 0, 0, 0, 0])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(
                weight,
                pre[:, :, t],
                post[:, :, t],
                reward=reward[t])

        assert weight.squeeze() > 1.00

    def test_homeostasis_dense_functional_updates_on_low_rate(self):
        """Test if the HomeostasisDense dynamics are correctly implemented."""
        update_rule = Homeostasis(1, 1, 1, tau=0.5, early_start=True)
        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]]])
        reward = torch.FloatTensor([0, 0, 0, 0, 0, 0])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(
                weight,
                pre[:, :, t],
                post[:, :, t],
                reward=reward[t])

        assert weight.squeeze() > 1.00

    def test_homeostasis_dense_functional_updates_on_high_rate(self):
        """Test if the HomeostasisDense dynamics are correctly implemented."""
        update_rule = Homeostasis(1, 1, 1, tau=0.5, early_start=True)
        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 1, 1, 0, 1, 1]]])
        reward = torch.FloatTensor([0, 0, 0, 0, 0, 0])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(
                weight,
                pre[:, :, t],
                post[:, :, t],
                reward=reward[t])

        assert weight.squeeze() < 1.00

    def test_homeostasis_dense_functional_works_on_batch(self):
        """Test if the HomeostasisDense update rule works on batch."""
        update_rule = Homeostasis(1, 1, 1, tau=0.5, T=5, early_start=True)

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
            self.fail("HomeostasisDense Update rule is not working on batch")

    def test_homeostasis_dense_functional_works(self):
        """DOC"""
        update_rule = Homeostasis(
            2, 2, 1,
            tau=0.5,
            window_size=4,
            early_start=False)

        pre = torch.FloatTensor([[
            [1, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0, 0]]])
        post = torch.FloatTensor([[
            [0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 1, 1, 0, 1, 1]]])
        reward = torch.FloatTensor([0, 0, 0, 0, 0, 0, 0, 0])

        weight = torch.Tensor([[1, 1], [1, 0]])

        for t in range(pre.shape[-1]):
            print(weight)
            weight = update_rule.update(
                weight,
                pre[:, :, t],
                post[:, :, t],
                reward=reward[t])

        assert weight[0, 0] > 1
        assert weight[1, 0] > 1
        assert weight[0, 1] < 1
        assert torch.abs(weight[1, 1]) < 0.01  # Assert 0
