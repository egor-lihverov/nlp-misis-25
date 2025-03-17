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
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.block_1 = SimpleBlock(hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.block_1(x)
        x = self.linear_2(x).squeeze(-1)
        return x


class SimpleNetV2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.block_1 = SimpleBlock(hidden_size)
        self.block_2 = SimpleBlock(hidden_size)
        self.block_3 = SimpleBlock(hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.linear_2(x).squeeze(-1)
        return x


class SimpleBlockV2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.bn = nn.BatchNorm1d(hidden_size)

        self.relu = nn.ReLU()

        self.linear = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        x = self.bn(x)

        identity = x

        x = self.relu(self.linear(x))
        x = self.output(x)

        x += identity
        return x


class SimpleNetV3(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.block_1 = SimpleBlockV2(hidden_size)
        self.block_2 = SimpleBlockV2(hidden_size)
        self.block_3 = SimpleBlockV2(hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.linear_2(x)
        return x.squeeze(-1)


class SimpleBlockV3(nn.Module):
    def __init__(self, hidden_size, p):
        super().__init__()
        self.drop_out = nn.Dropout(p=p)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        identity = x
        x = self.bn(x)
        x = self.relu(self.linear(x))
        x = self.output(x)
        x += identity
        x = self.drop_out(x)
        return x


class SimpleNetV4(nn.Module):
    def __init__(self, input_size, hidden_size, p):
        super().__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.block_1 = SimpleBlockV3(hidden_size, p)
        self.block_2 = SimpleBlockV3(hidden_size, p)
        self.block_3 = SimpleBlockV3(hidden_size, p)
        self.linear_2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.linear_2(x).squeeze(-1)
        return x
