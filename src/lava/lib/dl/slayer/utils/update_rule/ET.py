import torch

from .stdp import STDP_Base
from .functional import Compose, Identity


class ET:
    def __init__(
        self,
        in_neurons : int,
        out_neurons : int,
        e_decay : float = 0.85,
        e_alfa : float = 0.5
    ):

        self.e_decay = e_decay
        self.e_alfa = e_alfa

        self.e_trace = torch.zeros((out_neurons, in_neurons))

    def __call__(
        self,
        weight : torch.Tensor,
        pre : torch.Tensor,
        post : torch.Tensor,
        **kwargs : torch.Tensor
    ):
        if 'w_change' not in kwargs:
            raise ValueError("w_change not in kwargs")

        self.e_trace *= self.e_decay
        self.e_trace += self.e_alfa * kwargs['w_change']

        return self.e_trace.detach().clone()


class STDPET:
    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        batch_size : int,
        tau : float,
        beta : float,
        max_trace : float = -1.0,
        e_decay : float = 0.85,
        e_alfa : float = 0.5,
        nu_zero : float = 1,
    ):

        self.k = Compose()

        self.k.add(
            STDP_Base(
                in_neurons, out_neurons, batch_size,
                tau=tau, beta=beta, max_trace=max_trace,
                nu_zero=1))

        self.k.add(ET(in_neurons, out_neurons, e_decay, e_alfa))

        self.nu_zero = nu_zero

    def __call__(
        self,
        weight : torch.Tensor,
        pre : torch.Tensor,
        post : torch.Tensor,
        **kwargs
    ):

        return self.nu_zero * self.k(weight, pre, post, **kwargs)
