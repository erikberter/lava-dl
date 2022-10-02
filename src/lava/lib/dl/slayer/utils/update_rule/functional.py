import torch


class Identity:
    def __init__(self, param=None):
        self.param = param

    def __call__(self, **kwargs):
        if self.param is not None:
            return kwargs[self.param]
        else:
            return 1


class Compose:
    def __init__(self):
        self.functions = []

    def add(self, F):
        self.functions += [F]

    def __call__(
        self,
        weight : torch.Tensor,
        pre : torch.Tensor,
        post : torch.Tensor,
        **kwargs) -> torch.Tensor:

        w_change = self.functions[0](weight, pre, post)

        for f in self.functions[1:]:
            w_change = f(weight, pre, post, w_change=w_change, **kwargs)

        return w_change
