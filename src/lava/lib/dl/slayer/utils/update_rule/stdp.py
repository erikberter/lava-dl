import torch


class STDP_Functional:
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
            nu_zero : float = 0.01,
            tau : float = 0.92,
            beta : float = 1,
            A_plus : float = 1,
            A_minus : float = 1,
            max_trace : float = -1.0,
            use_weight_dependent : bool = False,
            use_nearest_neightboor : bool = False,
            **kwargs
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
        use_weight_dependent : bool
            True if the STDP will update weights based on a weight
            dependence.
        use_nearest_neightboor : bool
            True if the synaptic trace will be uptated on a neares neightboor
            based.
        """

        self.tau = tau
        self.beta = beta

        self.A_plus = A_plus
        self.A_minus = A_minus

        self.max_trace = max_trace

        self.nu_zero = nu_zero

        self.pre_trace = torch.zeros((batch_size, in_neurons))
        self.post_trace = torch.zeros((batch_size, out_neurons))

        self.use_weight_dependent = use_weight_dependent
        self.use_nearest_neightboor = use_nearest_neightboor

    def __call__(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """Updates synaptic traces. Also updates weights
        with GenericUpdateRule update function."""

        # Update synaptic trace
        if self.use_nearest_neightboor:
            self.pre_trace *= self.tau
            self.post_trace *= self.tau

            self.pre_trace = torch.max(self.beta * pre, self.pre_trace)
            self.post_trace = torch.max(self.beta * post, self.post_trace)
        else:
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

        if not self.use_weight_dependent:
            z = self.A_plus * A_plus_mat - self.A_minus * A_minus_mat
        else:
            w_dif = kwargs['W_max'] - kwargs['W_min']
            z = self.A_plus * A_plus_mat * (kwargs['W_max'] - weight) / w_dif
            z -= self.A_minus * A_minus_mat * (weight - kwargs['W_min']) / w_dif

        return self.nu_zero * z
