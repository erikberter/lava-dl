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
    def __init__(self,
            weight_decay : float = 1.0
        ) -> None:

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

