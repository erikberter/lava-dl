import torch

from .base import GenericUpdateRule

class HebbianDense(GenericUpdateRule):
    """
    Applies a hebbian weight update.
    
    Pre and Post should be in shape NA1 and NB1 respectively,
    where A and B are the features of pre and post.

    Parameters
    ----------
    mu : float, optional
        plasticity of the hebbian update rule.

    Attributes
    ----------
    mu : float
        plasticity of the hebbian update rule.
    """
    def __init__(self, mu : float = 0.05, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mu = mu
        
    def update(
            self,
            weight : torch.Tensor,
            pre : torch.Tensor,
            post : torch.Tensor) -> None:

        weight *= 1-self.mu
        weight += self.mu * torch.bmm(post, pre.permute(0, 2, 1))

        return weight
