import pytest
import torch
from src.exp_plus_cos import exp_plus_cos

@pytest.fixture
def setup_exp_plus_cos():
    X_val = torch.randn(3, 3, requires_grad=True)
    Y_val = torch.randn(3, 3, requires_grad=True)
    return X_val, Y_val

def test_exp_plus_cos_forward(setup_exp_plus_cos):
    X_val, Y_val = setup_exp_plus_cos
    X_custom = X_val.clone().detach().requires_grad_(True)
    Y_custom = Y_val.clone().detach().requires_grad_(True)
    X_pytorch = X_val.clone().detach().requires_grad_(True)
    Y_pytorch = Y_val.clone().detach().requires_grad_(True)

    output_custom = exp_plus_cos(X_custom, Y_custom)
    output_pytorch = torch.exp(X_pytorch) + torch.cos(Y_pytorch)

    assert torch.allclose(output_custom, output_pytorch, atol=1e-6), \
        f"Forward outputs differ. Max difference: {torch.abs(output_custom - output_pytorch).max()}"

def test_exp_plus_cos_backward(setup_exp_plus_cos):
    X_val, Y_val = setup_exp_plus_cos
    X_custom = X_val.clone().detach().requires_grad_(True)
    Y_custom = Y_val.clone().detach().requires_grad_(True)
    X_pytorch = X_val.clone().detach().requires_grad_(True)
    Y_pytorch = Y_val.clone().detach().requires_grad_(True)

    output_custom = exp_plus_cos(X_custom, Y_custom)
    loss_custom = output_custom.sum()
    loss_custom.backward()

    output_pytorch = torch.exp(X_pytorch) + torch.cos(Y_pytorch)
    loss_pytorch = output_pytorch.sum()
    loss_pytorch.backward()

    assert torch.allclose(X_custom.grad, X_pytorch.grad, atol=1e-6), \
        f"Gradients dL/dX differ. Max difference: {torch.abs(X_custom.grad - X_pytorch.grad).max()}"
    assert torch.allclose(Y_custom.grad, Y_pytorch.grad, atol=1e-6), \
        f"Gradients dL/dY differ. Max difference: {torch.abs(Y_custom.grad - Y_pytorch.grad).max()}"