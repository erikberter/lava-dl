import torch
from lava.lib.dl.slayer.utils.update_rule.base import GenericSTDPLearningRule
from lava.lib.dl.slayer.utils.update_rule.functional import Identity, Compose
from lava.lib.dl.slayer.utils.update_rule.mstdp import MSTDP_Functional


class StreamingMovingAverage:
    def __init__(self, n_out, window_size, early_start=False):
        self.window_size = window_size
        self.values = torch.zeros((window_size, n_out))
        self.early_start = early_start
        self.i = 0

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
            return torch.mean(self.values[:self.i], dim=0)

        return torch.mean(self.values, dim=0)

    def rate_available(self):
        return self.early_start or self.i >= self.window_size


class Homo:
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
        early_start : bool = False
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.r_exp = r_exp
        self.T = T

        self.sma = StreamingMovingAverage(
            n_out=out_neurons,
            window_size=window_size,
            early_start=early_start)

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

        w_weight = self.alpha * (1 - rate / self.r_exp) * weight

        w_weight = w_weight + w_change

        denominator = torch.abs(1 - rate / self.r_exp) * self.gamma
        # denominator *= self.T  I think this variable is not needed

        return rate / denominator * w_weight


class Homeostasis(GenericSTDPLearningRule):
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
        T : int = 50,
        window_size : int = 5,
        early_start : bool = False
    ):

        F = MSTDP_Functional(
            in_neurons,
            out_neurons,
            batch_size,
            tau,
            beta,
            max_trace,
            e_decay,
            e_alfa)

        G = Homo(
            in_neurons,
            out_neurons,
            batch_size,
            T=T,
            window_size=window_size,
            early_start=early_start)

        H = Compose()

        H.add(F)
        H.add(G)

        super().__init__(H, Identity())
