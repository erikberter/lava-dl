import torch
from abc import ABC


class GenericUpdateRule(ABC):
    """Abstract Update Rule class. Must be parent class of all update rules.
        Contains the dynamics which all update rules share.

        Attributes
        ----------
        weight_decay : float
            rate of decrease in weights
    """
    def __init__(self, weight_decay : float = 1.0) -> None:

        self.weight_decay = weight_decay

    def update(self, weight : torch.Tensor, **kwargs) -> torch.Tensor:
        """Updates the input weights.

        Parameters
        ----------
        weight : torch.Tensor
            Input weight of the layer

        Returns
        -------
        torch.Tensor.
            Updated weights.
        """

        weight *= self.weight_decay

        return weight


class DenseSynapticTraceUpdateRule(GenericUpdateRule):
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
            **kwargs):
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
        super().__init__(**kwargs)

        self.tau = tau
        self.beta = beta

        self.max_trace = max_trace

        self.pre_trace = torch.zeros((batch_size, in_neurons))
        self.post_trace = torch.zeros((batch_size, out_neurons))

    def update(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs) -> torch.Tensor:
        """Updates synaptic traces. Also updates weights
        with GenericUpdateRule update function."""

        weight = super().update(weight)

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

        return weight
