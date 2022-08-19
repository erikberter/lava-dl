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
            input : torch.Tensor,
            num_steps : int = 20):
        """Encodes the data on a rate encoding paradigm. Input should
        be in the range [0, 1].

        Expects data with a time a shape of [NCT], or, in gneeral, [N...T].

        Parameters
        ----------
        input : torch.Tensor
            Input tensor clamped on the range [0, 1]
        num_steps : int
            Number of output time steps.
        """
        if torch.min(input) < 0.0 or torch.max(input) > 1.0:
            raise Exception("Input dada should be between 0 and 1.")

        if num_steps < 1:
            raise Exception("num_steps should be greater than 1.")

        if num_steps > 1:
            torch.stack(num_steps * [input], dim=-1)

        return torch.bernoulli(input)
