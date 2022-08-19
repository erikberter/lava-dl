import torch

from .base import GenericUpdateRule
from .base import DenseSynapticTraceUpdateRule


class LinearMSTDPDense(DenseSynapticTraceUpdateRule):
    # TODO Change this doc
    """Applies a two-factor linear weight dependent stdp update.
    This implies that A_{+,i,j} and A_{-,i,j} are both constant.

    Attributes
    ----------
    tau : float
        Rate of decay of the synaptic trace.
    beta : float
        Change in the synaptic trace after each spike.
    A_plus : float
        Max change in STDP weight strengthening.
    A_minus : float
        Max change in STDP weight weakening.
    W_max : float
        Max value for a weight
    W_min : float
        Min value for a weight
    nu_zero : float
        Learning rate
    pre_trace : torch.Tensor
        Synaptic trace of input spikes.
    post_trace : torch.Tensor
        Synaptic trace of output spikes.
    """
    def __init__(
            self,
            in_neurons: int,
            out_neurons: int,
            batch_size : int,
            tau: float = 0.92,
            beta : float = 1.0,
            A_plus : float = 0.5,
            A_minus : float = 0.5,
            W_max : float = 2.0,
            W_min : float = 0.0,
            nu_zero : float = 0.1,
            **kwargs) -> None:
        """Initialization method.

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
        A_plus : float
            Max change in STDP weight strengthening.
        A_minus : float
            Max change in STDP weight weakening.
        W_max : float
            Max value for a weight
        W_min : float
            Min value for a weight
        nu_zero : float
            Learning rate
        """
        super(LinearMSTDPDense, self).__init__(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            batch_size=batch_size,
            tau=tau,
            beta=beta,
            **kwargs)

        self.nu_zero = nu_zero

        self.A_plus = A_plus
        self.A_minus = A_minus

        self.W_max = W_max
        self.W_min = W_min

    def update(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs) -> None:
        """Updates the weights based on the stdp dynamics"""

        if 'reward' not in kwargs:
            raise Exception("'reward' should be an argument.")

        # Start weight change calculation
        weight = super().update(weight, pre, post)

        # Apply Linear STDP dynamics
        A_plus_mat = torch.bmm(
            post.unsqueeze(dim=-1),
            torch.unsqueeze(self.pre_trace, dim=1)
        ).sum(dim=0)

        A_minus_mat = torch.bmm(
            torch.unsqueeze(self.post_trace, dim=-1),
            pre.unsqueeze(dim=1)
        ).sum(dim=0)

        self.A_plus_save = self.A_plus * A_plus_mat
        self.A_neg_save = self.A_minus * A_minus_mat

        z = self.A_plus * A_plus_mat - self.A_minus * A_minus_mat

        w_change = self.nu_zero * z

        weight += float(kwargs['reward']) * w_change

        # Clamp to bounds
        weight = torch.clamp(weight, self.W_min, self.W_max)

        return weight


class LinearMSTDPETDense(DenseSynapticTraceUpdateRule):
    # TODO Change this doc
    """Applies a two-factor linear weight dependent stdp update.
    This implies that A_{+,i,j} and A_{-,i,j} are both constant.

    Attributes
    ----------
    tau : float
        Rate of decay of the synaptic trace.
    beta : float
        Change in the synaptic trace after each spike.
    A_plus : float
        Max change in STDP weight strengthening.
    A_minus : float
        Max change in STDP weight weakening.
    W_max : float
        Max value for a weight
    W_min : float
        Min value for a weight
    e_decay : float
        Decay of the elasticity trace
    e_alfa : float
        Value added for each STDP step
    e_trace : torch.Tensor
        Elegibility trace
    pre_trace : torch.Tensor
        Synaptic trace of input spikes
    post_trace : torch.Tensor
        Synaptic trace of output spikes
    nu_zero : float
        Learning rate
    """
    def __init__(
            self,
            in_neurons: int,
            out_neurons: int,
            batch_size : int,
            tau: float = 0.92,
            beta : float = 1.0,
            A_plus : float = 0.5,
            A_minus : float = 0.5,
            W_max : float = 2.0,
            W_min : float = 0.0,
            e_decay : float = 0.85,
            e_alfa : float = 0.5,
            nu_zero : float = 0.1,
            **kwargs) -> None:
        """Initialization method.

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
        A_plus : float
            Max change in STDP weight strengthening.
        A_minus : float
            Max change in STDP weight weakening.
        W_max : float
            Max value for a weight
        W_min : float
            Min value for a weight
        e_decay : float
            Decay of the elasticity trace
        e_alfa : float
            Value added for each STDP step
        nu_zero : float
            Learning rate
        """
        super(LinearMSTDPETDense, self).__init__(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            batch_size=batch_size,
            tau=tau,
            beta=beta,
            **kwargs)

        self.e_decay = e_decay
        self.e_alfa = e_alfa

        self.nu_zero = nu_zero

        self.e_trace = torch.zeros((out_neurons, in_neurons))

        self.A_plus = A_plus
        self.A_minus = A_minus

        self.W_max = W_max
        self.W_min = W_min

    def update(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs) -> None:
        """Updates the weights based on the stdp dynamics"""

        if 'reward' not in kwargs:
            raise Exception("'reward' should be an argument.")

        # Start weight change calculation
        weight = super().update(weight, pre, post)

        # Apply Linear STDP dynamics
        A_plus_mat = torch.bmm(
            post.unsqueeze(dim=-1),
            torch.unsqueeze(self.pre_trace, dim=1)
        ).sum(dim=0)

        A_minus_mat = torch.bmm(
            torch.unsqueeze(self.post_trace, dim=-1),
            pre.unsqueeze(dim=1)
        ).sum(dim=0)

        z = self.A_plus * A_plus_mat - self.A_minus * A_minus_mat

        self.e_trace *= self.e_decay
        self.e_trace += self.e_alfa * z

        w_change = self.nu_zero * self.e_trace

        weight += float(kwargs['reward']) * w_change

        # Clamp to bounds
        weight = torch.clamp(weight, self.W_min, self.W_max)

        return weight
