# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Updatable synapse module"""

import numpy as np
import torch
import torch.nn.functional as F


class GenericUpdatableLayer(torch.nn.Module):
    """Abstract updatable synapse layer class.

    Attributes
    ----------
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(self):
        super(GenericUpdatableLayer, self).__init__()
        self.weight_norm_enabled = False
        self.complex = False
        self._update_rule = None

    def enable_weight_norm(self):
        """Enables weight normalization on synapse."""
        self = torch.nn.utils.weight_norm(self, name='weight')
        self.weight_norm_enabled = True

    def disable_weight_norm(self):
        """Disables weight normalization on synapse."""
        torch.nn.utils.remove_weight_norm(self, name='weight')
        self.weight_norm_enabled = False

    @property
    def grad_norm(self):
        """Norm of weight gradients. Useful for monitoring gradient flow."""
        if self.weight_norm_enabled is False:
            if self.weight.grad is None:
                return 0
            else:
                return torch.norm(
                    self.weight.grad
                ).item() / torch.numel(self.weight.grad)
        else:
            if self.weight_g.grad is None:
                return 0
            else:
                return torch.norm(
                    self.weight_g.grad
                ).item() / torch.numel(self.weight_g.grad)

    @property
    def pre_hook_fx(self):
        """Returns the pre-hook function for synapse operation. Typically
        intended to define the quantization method."""
        return self._pre_hook_fx

    @pre_hook_fx.setter
    def pre_hook_fx(self, fx):
        """Sets the pre-hook function for synapse operation. Typically intended
        to define the quantization method.
        """
        self._pre_hook_fx = fx

    @property
    def shape(self):
        """Shape of the synapse"""
        return self.weight.shape