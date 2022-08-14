# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import sys
import os
import unittest
import h5py

import numpy as np
import torch

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
    'persistent_state' : True}


class TestCUBA(unittest.TestCase):
    """Test CUBA blocks"""

    def test_dense_block(self):
        """Test dense block with lava process implementation."""
        in_features = 10
        out_features = 5
        time_steps = 10

        # create slayer network and evaluate output
        net = slayer.block.cuba.Dense(neuron_param, in_features, out_features)
        x = (torch.rand([1, in_features, time_steps]) > 0.5).float()
        y = net(x)
        
        # export slayer network
        net.export_hdf5(h5py.File(tempdir + '/cuba_dense.net',
                                  'w').create_group('layer/0'))

        # create equivalent lava network using netx and evaluate output
        lava_net = netx.hdf5.Network(net_config=tempdir + '/cuba_dense.net')
        source = io.source.RingBuffer(data=x[0])
        sink = io.sink.RingBuffer(shape=lava_net.out.shape, buffer=time_steps)
        source.s_out.connect(lava_net.inp)
        lava_net.out.connect(sink.a_in)
        run_condition = RunSteps(num_steps=time_steps)
        run_config = Loihi1SimCfg(select_tag='fixed_pt')
        lava_net.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        lava_net.stop()

        if verbose:
            print()
            print(lava_net)
            print('slayer output:')
            print(y[0])
            print('lava output:')
            print(output)

        self.assertTrue(np.abs(y[0].data.numpy() - output).sum() == 0)

    def test_conv_block(self):
        """Test conv block with lava process implementation."""
        height = 16
        width = 24
        in_features = 3
        out_features = 5
        kernel_size = 3
        time_steps = 10

        # create slayer network and evaluate output
        net = slayer.block.cuba.Conv(neuron_param,
                                     in_features, out_features, kernel_size)
        x = (torch.rand([1, in_features,
                         height, width, time_steps]) > 0.5).float()
        y = net(x).permute((0, 3, 2, 1, 4))

        # export slayer network
        net.export_hdf5(h5py.File(tempdir + '/cuba_conv.net',
                                  'w').create_group('layer/0'))

        # create equivalent lava network using netx and evaluate output
        lava_net = netx.hdf5.Network(net_config=tempdir + '/cuba_conv.net',
                                     input_shape=(width, height, in_features))
        source = io.source.RingBuffer(data=x[0].permute((2, 1, 0, 3)))
        sink = io.sink.RingBuffer(shape=lava_net.out.shape, buffer=time_steps)
        source.s_out.connect(lava_net.inp)
        lava_net.out.connect(sink.a_in)
        run_condition = RunSteps(num_steps=time_steps)
        run_config = Loihi1SimCfg(select_tag='fixed_pt')
        lava_net.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        lava_net.stop()

        if verbose:
            print()
            print(lava_net)
            print('slayer output:')
            print(y[0][0, 0, 0])
            print('lava output:')
            print(output[0, 0, 0])

        self.assertTrue(np.abs(y[0].data.numpy() - output).sum() == 0)

    def test_updatable_dense_block_contains_update_rule(self):
        """Test updatable dense contains update rule."""

        in_features = 10
        out_features = 5

        def update(weight, pre, post):
            return weight

        net = slayer.block.cuba.UpdatableDense(
            updatable_neuron_param,
            in_features,
            out_features,
            update_rule=update
        )

        assert net.synapse.update_rule is not None

    def test_updatable_dense_block_persists_state(self):
        """Test updatable dense mantains persists state on neurons."""

        # GIVEN
        in_features = 10
        out_features = 5
        time_steps = 1

        def update(weight, pre, post):
            return weight

        net = slayer.block.cuba.UpdatableDense(
            updatable_neuron_param,
            in_features,
            out_features,
            update_rule=update
        )

        x = (torch.ones([1, in_features, time_steps]) > 0.5).float()

        f_res = net(x).clone().detach()

        all_equal = False

        # WHEN
        for i in range(10):
            res = net(x).clone().detach()
            all_equal = all_equal and torch.equal(res, f_res)

        # THEN
        assert not all_equal

    def test_updatable_dense_block_returns_correct_shape(self):
        """Test updatable dense returns correct shape on execution."""

        # GIVEN
        in_features = 10
        out_features = 5
        batch_size = 3
        time_steps = 7

        def update(weight, pre, post):
            return weight

        net = slayer.block.cuba.UpdatableDense(
            updatable_neuron_param,
            in_features,
            out_features,
            update_rule=update
        )

        x = (torch.ones([batch_size, in_features, time_steps]) > 0.5).float()

        # WHEN
        res = net(x).clone().detach()

        # THEN
        assert res.shape == (3, 5, 7)

    def test_updatable_dense_block_with_class_based_update_rule(self):
        """Test updatable dense works with GenericUpdateRule subclasss."""

        # GIVEN
        in_features = 10
        out_features = 5
        batch_size = 3
        time_steps = 7

        from lava.lib.dl.slayer.utils.update_rule.base import GenericUpdateRule
        class CustomUpdateRule(GenericUpdateRule):
            def __init__(self):
                super(CustomUpdateRule, self).__init__()

            def update(self, weight, **kwargs):
                return weight

        update_rule = CustomUpdateRule()

        net = slayer.block.cuba.UpdatableDense(
            updatable_neuron_param,
            in_features,
            out_features,
            update_rule=update_rule
        )

        x = (torch.ones([batch_size, in_features, time_steps]) > 0.5).float()

        # WHEN
        net(x).clone().detach()
        assert True

    def test_updatable_dense_block_with_update_rule_passes_extra_data(self):
        """ Test updatable dense returns correct shape on execution. """

        # GIVEN
        in_features = 10
        out_features = 5
        batch_size = 3
        time_steps = 1

        from lava.lib.dl.slayer.utils.update_rule.base import GenericUpdateRule
        class CustomUpdateRule(GenericUpdateRule):
            def __init__(self):
                super(CustomUpdateRule, self).__init__()

            def update(self, weight, **kwargs):
                if 'reward' not in kwargs:
                    raise ValueError("No reward in parameters.")
                return weight

        update_rule = CustomUpdateRule()

        net = slayer.block.cuba.UpdatableDense(
            updatable_neuron_param,
            in_features,
            out_features,
            update_rule=update_rule
        )

        x = (torch.ones([batch_size, in_features, time_steps]) > 0.5).float()

        # WHEN
        try:
            net(x, reward=[0, -1, 2]).clone().detach()
        except ValueError:
            self.fail("Value error thrown.")

    # TODO Test updatable dense export to h5py.
