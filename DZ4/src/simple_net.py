import torch.nn as nn

class SimpleBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.output(x)
        return x

class SimpleNetV1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.block_1 = SimpleBlock(hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.block_1(x)
        x = self.linear_2(x).squeeze(-1)
        return x