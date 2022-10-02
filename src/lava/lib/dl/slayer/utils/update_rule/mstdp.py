import torch
from lava.lib.dl.slayer.utils.update_rule.base import GenericSTDPLearningRule
from lava.lib.dl.slayer.utils.update_rule.ET import STDPET
from lava.lib.dl.slayer.utils.update_rule.functional import Identity


class MSTDP(GenericSTDPLearningRule):
    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        batch_size : int,
        tau : float = 0.92,
        beta : float = 1,
        max_trace : float = -1.0,
        e_decay : float = 0.85,
        e_alfa : float = 0.5,
        nu_zero : float = 0.01,
    ):

        F = STDPET(
            in_neurons,
            out_neurons,
            batch_size,
            tau,
            beta,
            max_trace,
            e_decay,
            e_alfa,
            nu_zero=nu_zero)

        G = Identity('reward')

        super().__init__(F, G)


class MSTDP_Functional:
    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        batch_size : int,
        tau : float = 0.92,
        beta : float = 1,
        max_trace : float = -1.0,
        e_decay : float = 0.85,
        e_alfa : float = 0.5,
        nu_zero : float = 0.01,
    ):

        self.F = STDPET(
            in_neurons,
            out_neurons,
            batch_size,
            tau,
            beta,
            max_trace,
            e_decay,
            e_alfa,
            nu_zero=nu_zero)

        self.G = Identity('reward')

    def __call__(
        self,
        weight,
        pre,
        post,
        **kwargs
    ):

        return self.F(weight, pre, post, **kwargs) * self.G(**kwargs)
