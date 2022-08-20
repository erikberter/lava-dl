"""Spiking encoder class

    Encondings will be based on the definitions in
    ''Neural Coding in Spiking Neural Networks: A Comparative
    Study for Robust Neuromorphic Systems''
"""

import numpy as np
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod


class GenericEncoder(ABC):

    def __call__(self, input, num_steps : int = 1):
        return self.encode(input, num_steps)

    @abstractmethod
    def encode(self, input, num_steps : int = 1):
        pass


class RateEncoder(GenericEncoder):

    def encode(
            self,
            data : torch.Tensor,
            num_steps : int = 20):
        """Encodes the data on a rate encoding paradigm. Input should
        be in the range [0, 1].

        Expects data with a shape of [NC], or, in general, [N...C_k].

        Parameters
        ----------
        data : torch.Tensor
            Input tensor clamped on the range [0, 1].
        num_steps : int
            Number of output time steps.
        """
        if torch.min(data) < 0.0 or torch.max(data) > 1.0:
            raise ValueError("Input dada should be between 0 and 1.")

        if num_steps < 1:
            raise ValueError("num_steps should be greater than 1.")

        if num_steps > 1:
            data = torch.stack(num_steps * [data], dim=-1)

        return torch.bernoulli(data)


class TTFEncoder(GenericEncoder):
    """Time-to-first Encoder"""

    def encode(
            self,
            data : torch.Tensor,
            num_steps : int = 20):
        """Encodes the data on a time-to-first encoding paradigm. Input should
        be in the range [0, 1].

        Expects data with a time a shape of [NC], or, in general, [N...C_k].

        Parameters
        ----------
        data : torch.Tensor
            Input tensor clamped on the range [0, 1].
        num_steps : int
            Number of output time steps.
        """
        if torch.min(data) < 0.0 or torch.max(data) > 1.0:
            raise ValueError("Input dada should be between 0 and 1.")

        if len(data.shape) < 2:
            raise ValueError("Input data should have at least 2 dimensions.")

        if num_steps < 1:
            raise ValueError("num_steps should be greater than 1.")

        if num_steps > 1:
            data = torch.stack(num_steps * [data], dim=-1)

        # Get rate based encoding
        x = torch.bernoulli(data)

        # Swap time dimension to first place
        x = x.transpose(0, len(x.shape) - 1)

        # Create a valid spike mask
        valid_spike_mask = torch.ones_like(x[0])

        for t in range(x.shape[0]):
            # Choose only valid spikes
            x[t] *= torch.where(x[t] > 0, valid_spike_mask, x[t])

            # Invalidate current valid spikes for the future
            valid_spike_mask -= x[t]

        x = x.transpose(0, len(x.shape) - 1)

        return x
