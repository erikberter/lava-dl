
"""Leaky Integrator module"""

import numpy as np
import torch
import torch.nn.functional as F


class LeakyIntegrator(torch.nn.Module):
    """Leaky Integrator to store current."""

    def __init__(
            self,
            n_input : int,
            batch_size : int,
            decay : float = 0.85,
            gamma : float = 0.6):
        """Initialation method for the Leaky Integrator."""
        super(LeakyIntegrator, self).__init__()
        self.decay = decay

        self.gamma = gamma

        self.state = torch.zeros((batch_size, n_input))

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        """Returns the updated leaky intergator state."""
        self.state *= self.decay
        self.state += self.gamma * input.squeeze(dim=2)

        return self.state.clone().detach()
