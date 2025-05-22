import pytest
import torch
import torch.nn as nn
from src.rmsnorm import RMSNorm

@pytest.fixture
def setup_rmsnorm():
    batch_size = 2
    seq_len = 3
    hidden_dim = 4
    epsilon_val = 1e-5
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    return batch_size, seq_len, hidden_dim, epsilon_val, input_tensor

def test_rmsnorm_forward(setup_rmsnorm):
    batch_size, seq_len, hidden_dim, epsilon_val, input_tensor = setup_rmsnorm

    custom_rmsnorm = RMSNorm(dims=hidden_dim, eps=epsilon_val)
    custom_rmsnorm.scale.data = torch.rand(hidden_dim)
    pytorch_rmsnorm = nn.RMSNorm(normalized_shape=hidden_dim, eps=epsilon_val)
    pytorch_rmsnorm.weight.data = custom_rmsnorm.scale.data.clone()

    output_custom = custom_rmsnorm(input_tensor.clone())
    output_pytorch = pytorch_rmsnorm(input_tensor.clone())

    assert torch.allclose(output_custom, output_pytorch, atol=1e-6), \
        f"Outputs differ. Max difference: {torch.abs(output_custom - output_pytorch).max()}"

def test_rmsnorm_forward_scale_ones(setup_rmsnorm):
    batch_size, seq_len, hidden_dim, epsilon_val, input_tensor = setup_rmsnorm

    custom_rmsnorm = RMSNorm(dims=hidden_dim, eps=epsilon_val)
    pytorch_rmsnorm = nn.RMSNorm(normalized_shape=hidden_dim, eps=epsilon_val)

    output_custom = custom_rmsnorm(input_tensor.clone())
    output_pytorch = pytorch_rmsnorm(input_tensor.clone())

    assert torch.allclose(output_custom, output_pytorch, atol=1e-6), \
        f"Outputs with scale=1 differ. Max difference: {torch.abs(output_custom - output_pytorch).max()}"