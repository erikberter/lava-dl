# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch

from lava.lib.dl.slayer.utils.update_rule.stdp import STDP_Functional
from lava.lib.dl.slayer.utils.update_rule.functional import Identity
from lava.lib.dl.slayer.utils.update_rule.base import GenericSTDPLearningRule
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


class TestLinearSTDPDenseFunctional(unittest.TestCase):
    """Test LinearSTDPDense blocks"""

    def test_linear_stdp_dense_functional_creation(self):
        """Test if the LinearSTDP class can be correctly launched."""
        try:
            F = STDP_Functional(1, 1, 1)
            update_rule = GenericSTDPLearningRule(F, Identity())
        except Exception:
            self.fail("LinearSTDPDense creation failed")

    def test_linear_stdp_dense_functional_updates_correctly(self):
        """Test if the STDP dynamics are correctly implemented."""
        F = STDP_Functional(1, 1, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]]])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.equal(weight, torch.FloatTensor([[0.9264]]))

    def test_linear_stdp_works_on_batch(self):
        """Test if the STDP update rule works on batch."""
        F = STDP_Functional(1, 1, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0]], [[1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]], [[1, 0, 0, 1, 0, 0]]])

        weight = torch.Tensor([[1]])

        try:
            for t in range(pre.shape[-1]):
                weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])
        except Exception:
            self.fail("STDP update rule does not work on batch.")

    def test_linear_stdp_does_not_create_weight(self):
        """DOC"""
        F = STDP_Functional(2, 2, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0]]])

        weight = torch.Tensor([[1, 1], [1, 0]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.equal(
            weight,
            torch.FloatTensor([[0.9264, 0.9264], [0.9264, 0.0]]))

    def test_linear_stdp_does_not_create_weight_2(self):
        """DOC"""
        F = STDP_Functional(2, 2, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]])

        weight = torch.Tensor([[1, 1], [1, 0]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.equal(
            weight,
            torch.FloatTensor([[1.0736, 1.0736], [1.0736, 0.0]]))

    def test_linear_stdp_does_not_create_weight_different_shape(self):
        """DOC"""
        F = STDP_Functional(2, 1, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0]]])
        post = torch.FloatTensor([[[0, 1, 0, 0, 0, 0]]])

        weight = torch.Tensor([[1, 0]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.equal(
            weight,
            torch.FloatTensor([[1.0736, 0.0]]))

    def test_linear_stdp_dense_functional_does_not_change_sign(self):
        """Test if the STDP dynamics are correctly implemented."""
        F = STDP_Functional(1, 1, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[0, 1, 0, 0, 0, 1]]])
        post = torch.FloatTensor([[[1, 0, 0, 0, 1, 0]]])

        weight = torch.Tensor([[1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.equal(weight, torch.FloatTensor([[0.0]]))

    def test_linear_stdp_dense_functional_updates_correctly_on_antihebbian(self):
        """Test if the STDP dynamics are correctly implemented."""
        F = STDP_Functional(1, 1, 1, nu_zero=1)
        update_rule = GenericSTDPLearningRule(F, Identity())

        pre = torch.FloatTensor([[[1, 0, 0, 0, 0, 0]]])
        post = torch.FloatTensor([[[0, 0, 1, 0, 0, 0]]])

        weight = torch.Tensor([[-1]])
        for t in range(pre.shape[-1]):
            weight = update_rule.update(weight, pre[:, :, t], post[:, :, t])

        assert torch.isclose(weight, torch.FloatTensor([[-0.1536]]), rtol=1e-3)
