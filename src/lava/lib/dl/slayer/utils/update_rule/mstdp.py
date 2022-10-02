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
        **kwargs
    ):

        F = STDPET(
            in_neurons,
            out_neurons,
            batch_size,
            **kwargs)

        G = Identity('reward')

        super().__init__(F, G)


class MSTDP_Functional:
    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        batch_size : int,
        **kwargs
    ):

        self.F = STDPET(
            in_neurons,
            out_neurons,
            batch_size,
            **kwargs)

        self.G = Identity('reward')

    def __call__(
        self,
        weight,
        pre,
        post,
        **kwargs
    ):

        return self.F(weight, pre, post, **kwargs) * self.G(**kwargs)
