import torch


class GenericSTDPLearningRule:
    def __init__(
        self,
        F,
        G,
        H=None,
        W_max=2.0,
        W_min=0.0,
        weight_norm=None,
        synaptogenesis=None,
        weight_decay=1.0,
        **kwargs
    ):
        self.F = F
        self.G = G
        self.H = H

        self.W_max = W_max
        self.W_min = W_min

        self.weight_decay = weight_decay

        self.weight_norm = weight_norm
        self.synaptogenesis = synaptogenesis

    def update(self, weight, pre, post, **kwargs):

        kwargs['W_max'] = self.W_max
        kwargs['W_min'] = self.W_min

        weight_copy = weight.clone().detach()

        zero_mask = weight != 0
        sign = weight.sign()

        w_change = self.F(weight, pre, post, **kwargs) * self.G(**kwargs)

        if self.H is not None:
            w_change += self.H(**kwargs)

        w_change *= zero_mask
        weight += w_change

        # Signed Clamp
        weight *= sign
        weight = weight.clamp_(self.W_min, self.W_max)
        weight *= sign

        if self.weight_norm is not None:
            weight = torch.nn.functional.normalize(weight, p=1.0, dim=1)
            weight *= self.weight_norm

        if self.synaptogenesis is not None:
            import random

            if random.randint(0, 300) == 0:
                a = torch.zeros_like(weight).uniform_(0, 1) * 0.05

                a *= ~zero_mask

                w_change = torch.bernoulli(a) * self.synaptogenesis

                weight += w_change

        weight = self.weight_decay * weight

        return weight

    def __call__(self, weight, pre, post, **kwargs):
        return self.update(weight, pre, post, **kwargs)
