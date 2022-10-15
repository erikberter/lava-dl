import torch
from lava.lib.dl.slayer.utils.update_rule.base import GenericSTDPLearningRule
from lava.lib.dl.slayer.utils.update_rule.functional import Identity, Compose
from lava.lib.dl.slayer.utils.update_rule.mstdp import MSTDP_Functional


class StreamingMovingAverage:
    def __init__(
        self,
        n_out,
        window_size,
        early_start=False,
        min_val : float = 0.1
    ):
        self.window_size = window_size
        self.values = torch.zeros((window_size, n_out))
        self.early_start = early_start
        self.i = 0

        self.min_val = min_val

    def __call__(self, value):
        if self.i < self.window_size:
            b_value = torch.mean(value, dim=0).clone().detach().squeeze()
            self.values[self.i] = b_value

            self.i += 1
        else:
            b_value = torch.mean(value, dim=0).clone().detach().squeeze()

            self.values = torch.roll(self.values, 1, 0)
            self.values[-1] = b_value

        return self.get_rate()

    def get_rate(self):
        if self.i < self.window_size:
            val = torch.mean(self.values[:self.i], dim=0)
        else:
            val = torch.mean(self.values, dim=0)

        return torch.maximum(
            val,
            self.min_val * torch.ones_like(val)
        )

    def rate_available(self):
        return self.early_start or self.i >= self.window_size


class Homeostasis_Functional:
    """
    Based on Biologically Plausible Models of
    Homeostasis and STDP: Stability and Learning
    in Spiking Neural Networks
    """
    def __init__(
        self,
        in_neurons,
        out_neurons,
        batch_size,
        r_exp : float = 0.3,
        T : int = 50,
        gamma : float = 1.0,
        alpha : float = 0.02,
        window_size : int = 50,
        early_start : bool = False,
        min_val : float = 0.1,
        **kwargs
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.r_exp = r_exp
        self.T = T

        self.w_change_s = 0.0
        self.w_update_s = 0.0

        self.sma = StreamingMovingAverage(
            n_out=out_neurons,
            window_size=window_size,
            early_start=early_start,
            min_val=min_val)

    def __call__(
        self,
        weight,
        pre,
        post,
        **kwargs
    ):
        if not self.sma.rate_available():
            rate = self.sma(post)

            return kwargs["w_change"]

        w_change = kwargs["w_change"]
        rate = self.sma(post)

        w_weight = self.alpha * torch.abs(weight)
        w_weight *= (1 - rate / self.r_exp)[:, None]

        self.w_update_s += w_weight.abs().sum()
        self.w_change_s += w_change.abs().sum()

        w_weight = w_weight + w_change

        denominator = 1 + torch.abs(1 - rate / self.r_exp) * self.gamma
        # denominator *= self.T  I think this variable is not needed

        return (rate[:, None] / denominator[:, None]) * w_weight


class Homeostasis(GenericSTDPLearningRule):
    def __init__(
        self,
        in_neurons: int,
        out_neurons: int,
        batch_size : int,

        T : int = 50,
        window_size : int = 50,
        early_start : bool = False,
        min_val : float = 0.1,
        gamma : float = 1.0,
        **kwargs
    ):

        F = MSTDP_Functional(
            in_neurons,
            out_neurons,
            batch_size,
            **kwargs)

        G = Homeostasis_Functional(
            in_neurons,
            out_neurons,
            batch_size,
            T=T,
            gamma=gamma,
            window_size=window_size,
            early_start=early_start,
            min_val=min_val,
            **kwargs)

        H = Compose()

        H.add(F)
        H.add(G)

        super().__init__(H, Identity(), **kwargs)
