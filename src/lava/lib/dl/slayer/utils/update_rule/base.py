import torch


class GenericSTDPLearningRule:
    def __init__(self, F, G, H=None, W_max=2.0, W_min=0.0, **kwargs):
        self.F = F
        self.G = G
        self.H = H

        self.W_max = W_max
        self.W_min = W_min

    def update(self, weight, pre, post, **kwargs):

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

        return weight

    def __call__(self, weight, pre, post, **kwargs):
        return self.update(weight, pre, post, **kwargs)
