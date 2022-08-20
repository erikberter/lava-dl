import torch

from .base import GenericUpdateRule, DenseSynapticTraceUpdateRule


class LinearSTDPDense(DenseSynapticTraceUpdateRule):
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
    W_max : float
        Max value for a weight.
    W_min : float
        Min value for a weight.
    pre_trace : torch.Tensor
        Synaptic trace of input spikes.
    post_trace : torch.Tensor
        Synaptic trace of output spikes.
    nu_zero : float
        Learning rate.
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
            W_max : float = 2.0,
            W_min : float = 0.0,
            nu_zero : float = 0.01,
            **kwargs) -> None:
        """Initialization method.

        Parameters
        ----------
        in_neurons : int
            Number of input neurons.
        out_neurons : int
            Number of output neurons.
        batch_size : int
            Size of the batch.
        tau : float
            Rate of decay of the synaptic trace.
        beta : float
            Change in the synaptic trace after each spike.
        A_plus : float
            Max change in STDP weight strengthening.
        A_minus : float
            Max change in STDP weight weakening.
        W_max : float
            Max value for a weight.
        W_min : float
            Min value for a weight.
        nu_zero : float
            Learning rate.
        """
        super(LinearSTDPDense, self).__init__(
            in_neurons=in_neurons,
            out_neurons=out_neurons,
            batch_size=batch_size,
            tau=tau,
            beta=beta,
            **kwargs)

        self.A_plus = A_plus
        self.A_minus = A_minus

        self.W_max = W_max
        self.W_min = W_min

        self.nu_zero = nu_zero

    def update(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs) -> None:
        """Updates the weights based on the stdp dynamics."""

        weight = super().update(weight, pre, post)

        # Get all the inhibitory neurons
        inhib_mask = torch.where(weight < 0, -1, 1)
        weight = torch.abs(weight)

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
        weight += self.nu_zero * z * inhib_mask

        # Clamp to zero
        weight = torch.clamp(weight, self.W_min, self.W_max)

        # Reapply inhib mask
        weight *= inhib_mask

        return weight
