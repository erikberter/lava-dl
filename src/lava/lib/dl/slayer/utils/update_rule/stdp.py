import torch

from .base import GenericUpdateRule


class LinearSTDPDense(GenericUpdateRule):
    """Applies a two-factor linear stdp update.
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
            A_plus : float = 1.0,
            A_minus : float = 1.0,
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
        """
        super(LinearSTDPDense, self).__init__(**kwargs)
        self.tau = tau
        self.beta = beta

        self.A_plus = A_plus
        self.A_minus = A_minus

        self.pre_trace = torch.zeros((batch_size, in_neurons))
        self.post_trace = torch.zeros((batch_size, out_neurons))

    def update(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs) -> None:
        """Updates the weights based on the stdp dynamics"""

        weight = super().update(weight)

        # Update synaptic trace
        self.pre_trace *= self.tau
        self.pre_trace += self.beta * pre

        self.post_trace *= self.tau
        self.post_trace += self.beta * post

        # Apply Linear STDP dynamics
        A_plus_mat = torch.bmm(
            post.unsqueeze(dim=-1),
            torch.unsqueeze(self.pre_trace, dim=1)
        ).sum(dim=0)

        A_minus_mat = torch.bmm(
            torch.unsqueeze(self.post_trace, dim=-1),
            pre.unsqueeze(dim=1)
        ).sum(dim=0)

        weight += self.A_plus * A_plus_mat - self.A_minus * A_minus_mat

        # Clamp to zero
        weight = torch.maximum(torch.zeros_like(weight), weight)

        return weight
