import pytest
import torch
import torch.nn as nn
from src.lion import Lion
from src.simple_net import SimpleNetV1

@pytest.fixture
def setup_lion():
    input_size = 10
    hidden_size = 20
    output_size = 1
    batch_size = 5
    lr = 3e-4
    betas = (0.9, 0.99)
    weight_decay = 0.01

    model = SimpleNetV1(input_size, hidden_size, output_size)
    optimizer = Lion(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    X_train = torch.randn(batch_size, input_size)
    y_train = torch.randn(batch_size, output_size)

    return model, optimizer, criterion, X_train, y_train

def test_lion_parameter_update(setup_lion):
    model, optimizer, criterion, X_train, y_train = setup_lion

    params_before = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    params_after = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    changed = any(not torch.equal(p_before, p_after) for p_before, p_after in zip(params_before, params_after))

    assert changed, "Parameters did not change after one Lion step."
    assert any(p.grad is not None for p in model.parameters()), "No gradients computed."

def test_lion_loss_decrease(setup_lion):
    model, optimizer, criterion, X_train, y_train = setup_lion

    initial_loss = float('inf')
    for epoch in range(10):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        assert current_loss < float('inf'), f"Loss is invalid at epoch {epoch + 1}"
        if epoch > 0:
            assert current_loss <= initial_loss, f"Loss did not decrease at epoch {epoch + 1}"
        initial_loss = current_loss