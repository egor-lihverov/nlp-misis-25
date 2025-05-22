import torch
from torch.autograd import Function

class ExpPlusCos(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        exp_X = torch.exp(X)
        cos_Y = torch.cos(Y)
        ctx.save_for_backward(X, Y)
        return exp_X + cos_Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X, Y = ctx.saved_tensors
        grad_X = grad_output * torch.exp(X)
        grad_Y = grad_output * -torch.sin(Y)
        return grad_X, grad_Y

def exp_plus_cos(X, Y):
    return ExpPlusCos.apply(X, Y)