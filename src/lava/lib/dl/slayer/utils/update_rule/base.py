from turtle import update
import torch
from abc import ABC


class GenericSTDPLearningRule:
    def __init__(self, F, G, H):
        self.F = F
        self.H = H
        self.G = G

    def update(self, weight, pre, post, **kwargs):
        w_change = self.F(weight, pre, post) * self.G(kwargs) + self.H(kwargs)

        weight += w_change

        # Signed Clamp
        sign = weight.sign()
        weight = weight.abs_().clamp_(self.W_min, self.W_max)
        weight *= sign

        return weight


class Identity:
    def __init__(self, param = None):
        self.param = param

    def __call__(self, **kwargs):
        if self.param is not None:
            return kwargs[self.param]
        else:
            return 1

class STDP_Base:
    """Abstract Update Rule class that implements synaptic trace
    for dense layers.

    This class should not be instanciated by itself.

    Attributes
    ----------
    tau : float
        Rate of decay of the synaptic trace.
    beta : float
        Change in the synaptic trace after each spike.
    pre_trace : torch.Tensor
        Synaptic trace of input spikes.
    post_trace : torch.Tensor
        Synaptic trace of output spikes.
    max_trace : float
        Defines the maximum value for the trace. If set to -1.0, then
        the trace will be unbounded."""

    def __init__(
            self,
            in_neurons: int,
            out_neurons: int,
            batch_size : int,
            tau : float,
            beta : float,
            max_trace : float = -1.0,
            ):
        """
        Initialization of Synaptic Trace based update rule.

        Parameters
        ----------
        in_neurons : int
            Number of input neurons
        out_neurons : int
            Number of output neurons
        batch_size : int
            Size of the batch
        tau : float
            Rate of decay of the synaptic trace.
        beta : float
            Change in the synaptic trace after each spike.
        max_trace : float
            Defines the maximum value for the trace. If set to -1.0, then
            the trace will be unbounded.
        """

        self.tau = tau
        self.beta = beta

        self.max_trace = max_trace

        self.pre_trace = torch.zeros((batch_size, in_neurons))
        self.post_trace = torch.zeros((batch_size, out_neurons))

    def __call__(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs) -> torch.Tensor:
        """Updates synaptic traces. Also updates weights
        with GenericUpdateRule update function."""

        # Update synaptic trace
        if self.max_trace == -1.0:
            # Unbounded synaptic trace
            self.pre_trace = self.tau * self.pre_trace + self.beta * pre
            self.post_trace = self.tau * self.post_trace + self.beta * post

        else:
            # Bounded synaptic trace
            pre_diff = self.beta * (self.max_trace - self.pre_trace) * pre
            self.pre_trace = self.tau * self.pre_trace + pre_diff

            post_diff = self.beta * (self.post_trace - self.pre_trace) * pre
            self.post_trace = self.tau * self.post_trace + post_diff

        A_plus_mat = torch.bmm(
            post.unsqueeze(dim=-1),
            torch.unsqueeze(self.pre_trace, dim=1)
        ).sum(dim=0)

        A_minus_mat = torch.bmm(
            torch.unsqueeze(self.post_trace, dim=-1),
            pre.unsqueeze(dim=1)
        ).sum(dim=0)

        z = self.A_plus * A_plus_mat - self.A_minus * A_minus_mat

        return z

class Compose:
    def __init__(self):
        self.functions = []

    def add(self, F):
        self.functions += [F]

    def __call__(self,
        weight : torch.Tensor,
        pre : torch.Tensor,
        post : torch.Tensor,
        **kwargs) -> torch.Tensor:
        
        w_change = self.functions[0](weight, pre, post)

        for f in self.functions[1:]:
            w_change = f(weight, pre, post, {"w_change" : w_change})
        
        return w_change

class ET:
    def __init__(self, 
            in_neurons: int,
            out_neurons: int,
            e_decay : float = 0.85,
            e_alfa : float = 0.5
            ):

        self.e_decay = e_decay
        self.e_alfa = e_alfa

        self.e_trace = torch.zeros((out_neurons, in_neurons))

    def __call__(self,
        weight : torch.Tensor,
        pre : torch.Tensor,
        post : torch.Tensor,
        w_change : torch.Tensors):

        self.e_trace *= self.e_decay
        self.e_trace += self.e_alfa * w_change

        return w_change

class STDPET:
    def __init__(self, 
        in_neurons: int,
        out_neurons: int,
        batch_size : int,
        tau : float,
        beta : float,
        max_trace : float = -1.0,
        e_decay : float = 0.85,
        e_alfa : float = 0.5
        ):

        self.k = Compose()
        self.k.add(STDP_Base(in_neurons, out_neurons, tau, beta, max_trace))
        self.k.add(ET(in_neurons, out_neurons, batch_size, e_decay, e_alfa))
    
    def __call__(self,
        weight : torch.Tensor,
        pre : torch.Tensor,
        post : torch.Tensor,
        **kwargs):

        return self.k(weight, pre, post, kwargs)

class MSTDP(GenericSTDPLearningRule):
    def __init__(self,
        in_neurons: int,
        out_neurons: int,
        batch_size : int,
        tau : float,
        beta : float,
        max_trace : float = -1.0,
        e_decay : float = 0.85,
        e_alfa : float = 0.5
        ):
        
        F = STDPET(in_neurons, out_neurons, batch_size, tau, beta, max_trace, e_decay, e_alfa)

        G = Identity('reward') 

        self.__init__(F, G, Identity())


