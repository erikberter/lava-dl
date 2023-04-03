# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Updatable synapse module"""

import numpy as np
import torch
import torch.nn.functional as F

from ..utils.update_rule.base import GenericSTDPLearningRule


class GenericUpdatableLayer(torch.nn.Module):
    """Abstract updatable synapse layer class.

    Attributes
    ----------
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    _update_rule : method
        method to apply after each time step to update weights.
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
    def update_rule(self):
        # TODO Doc function
        return self._update_rule

    @update_rule.setter
    def update_rule(self, update_rule):
        # TODO Doc function
        self._update_rule = update_rule

    @property
    def shape(self):
        """Shape of the synapse"""
        return self.weight.shape


class Dense(torch.nn.Linear, GenericUpdatableLayer):
    """Dense updatable synapse layer.

    Update rules can be either a subclass of
    GenericUpdateRule(<lava.lib.dl.slayer.utils.update_rule.base.GenericUpdateRule>)
    or a custom function.

    Parameters
    ----------
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    update_rule : method or GenericUpdateRule subclass
        Util to update weigths after each time step.
        Defaults to None.
    weight_scale : int
        weight initialization scaling factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Attributes
    ----------
    in_channels
    out_channels
    _update_rule : method or GenericUpdateRule subclass
        None.
    weight
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self,
        in_neurons,
        out_neurons,
        update_rule=None,
        weight_scale=1,
        weight_norm=False,
        pre_hook_fx=None,
        sparsity=1.0,
        inhib_ratio=0.2,
    ):
        """ """

        super(Dense, self).__init__(
            in_neurons, out_neurons, bias=False
        )

        self.update_rule = update_rule

        if weight_scale != 1:
            self.weight = torch.nn.Parameter(weight_scale * self.weight)

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

        self.weight = torch.nn.Parameter(
            torch.abs(self.weight), requires_grad=False)

        zero_mask = torch.rand(*self.weight.shape) < sparsity

        inhib_mask = torch.where(
            torch.rand(*self.weight.shape) < inhib_ratio, -1, 1)

        self.weight = torch.nn.Parameter(
            inhib_mask * zero_mask * self.weight,
            requires_grad=False
        )

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NC1, since only one time step is taken.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        N, C = input.shape[0], input.shape[1]

        out = super().forward(input.reshape((N, C)))
        return torch.unsqueeze(out, dim=-1)

    def apply_update_rule(self, **kwargs) -> None:
        """Applies the update rule on the weight."""

        if isinstance(self._update_rule, GenericSTDPLearningRule):
            self.weight = torch.nn.Parameter(
                self._update_rule.update(self.weight, **kwargs),
                requires_grad=False)
        else:
            self.weight = torch.nn.Parameter(
                self._update_rule(self.weight, **kwargs),
                requires_grad=False)


class LocallyConnected(GenericUpdatableLayer):
    """Locally Connected updatable synapse layer.

    Update rules can be either a subclass of
    GenericUpdateRule(<lava.lib.dl.slayer.utils.update_rule.base.GenericUpdateRule>)
    or a custom function.

    Parameters
    ----------
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    update_rule : method or GenericUpdateRule subclass
        Util to update weigths after each time step.
        Defaults to None.
    weight_scale : int
        weight initialization scaling factor. Defaults to 1.
    weight_norm : bool
        flag to enable/disable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function reference or a lambda function. If the function is provided,
        it will be applied to it's weight before the forward operation of the
        synapse. Typically the function is a quantization mechanism of the
        synapse. Defaults to None.

    Attributes
    ----------
    in_channels
    out_channels
    _update_rule : method or GenericUpdateRule subclass
        None.
    weight
    weight_norm_enabled : bool
        flag indicating weather weight norm in enabled or not.
    complex : bool
        False. Indicates synapse is not complex.
    """
    def __init__(
        self,
        in_neurons,
        out_neurons,
        kernel_size,
        stride=1,
        update_rule=None,
        weight_scale=1,
        weight_norm=False,
        pre_hook_fx=None,
        sparsity=1.0,
        inhib_ratio=0.2,
    ):
        """ """

        super(LocallyConnected, self).__init__(
            in_neurons, out_neurons, bias=False
        )

        self.update_rule = update_rule

        if weight_scale != 1:
            self.weight = torch.nn.Parameter(weight_scale * self.weight)

        self._pre_hook_fx = pre_hook_fx

        if weight_norm is True:
            self.enable_weight_norm()

        self.weight = torch.nn.Parameter(
            torch.abs(self.weight), requires_grad=False)

        zero_mask = torch.rand(*self.weight.shape) < sparsity

        inhib_mask = torch.where(
            torch.rand(*self.weight.shape) < inhib_ratio, -1, 1)

        self.weight = torch.nn.Parameter(
            inhib_mask * zero_mask * self.weight,
            requires_grad=False
        )

    def forward(self, input):
        """Applies the synapse to the input.

        Parameters
        ----------
        input : torch tensor
            Input tensor. Typically spikes. Input is expected to be of shape
            NC1, since only one time step is taken.

        Returns
        -------
        torch tensor
            dendrite accumulation / weighted spikes.

        """
        N, C = input.shape[0], input.shape[1]

        out = super().forward(input.reshape((N, C)))
        return torch.unsqueeze(out, dim=-1)

    def apply_update_rule(self, **kwargs) -> None:
        """Applies the update rule on the weight."""

        if isinstance(self._update_rule, GenericSTDPLearningRule):
            self.weight = torch.nn.Parameter(
                self._update_rule.update(self.weight, **kwargs),
                requires_grad=False)
        else:
            self.weight = torch.nn.Parameter(
                self._update_rule(self.weight, **kwargs),
                requires_grad=False)
