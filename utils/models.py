import torch.nn as nn
import torch
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        # self.relu1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class co_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(co_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
             nn.Sigmoid(),
             nn.Linear(output_dim, 64),
             nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        out_1 = self.co_block(out)
        out_1 = self.output(out_1)+out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class co_MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(co_MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
             nn.Sigmoid(),
             nn.Linear(output_dim, 64),
             nn.ELU(),
             nn.Linear(64, output_dim)
        )
        self.co_block_1 = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        out_1 = self.co_block(out)+out
        out_1 = self.co_block_1(out_1)
        out_1 = self.output(out_1)+out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MLPC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, output_dim)  # correlation matrix
        )

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out_Y = self.out(out)
        return out, out_Y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class co_linear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(co_linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 模型原始输出

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)

        out_1 = self.co_block(out)
        out_1 = self.output(out_1) + out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class co_linear2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(co_linear2, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # 模型原始输出

        # 自动学习标签关联性
        self.co_block = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
            nn.Linear(64, output_dim)
        )
        self.co_block_1 = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, 64),
            nn.ELU(),
        )
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)

        out_1 = self.co_block(out) + out
        out_1 = self.co_block_1(out_1)
        out_1 = self.output(out_1) + out
        return out_1

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class linearC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearC, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.out = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(output_dim, output_dim)  # correlation matrix
        )

    def forward(self, x):
        out = x.view(-1, self.num_flat_features(x))  # 把图片展成一维的
        out = self.linear(out)
        out_Y = self.out(out)
        return out, out_Y

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class GhostBatchNorm(nn.Module):
    def __init__(self, num_features, virtual_batch_size=128, momentum=0.01):
        super(GhostBatchNorm, self).__init__()
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(x.size(0) / self.virtual_batch_size) + 1, dim=0)
        res = [self.bn(chunk) for chunk in chunks]
        return torch.cat(res, dim=0)


class TabNetLayer(nn.Module):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128):
        super(TabNetLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.glu(x)
        return x


class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=16, n_step=3, virtual_batch_size=128):
        super(TabNet, self).__init__()
        self.initial_bn = GhostBatchNorm(input_dim, virtual_batch_size)
        self.first_layer = TabNetLayer(input_dim, 2 * hidden_dim, virtual_batch_size)
        self.tabnet_layers = nn.ModuleList([
            TabNetLayer(hidden_dim, 2 * hidden_dim, virtual_batch_size) for _ in range(n_step - 1)
        ])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.initial_bn(x)
        x = self.first_layer(x)
        for layer in self.tabnet_layers:
            x = layer(x)
        x = self.fc(x)
        return x
