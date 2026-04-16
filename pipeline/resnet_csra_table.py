import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
import torch.utils.model_zoo as model_zoo
import logging

# 预训练模型的URLs
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class CSRA(nn.Module):
    def __init__(self, input_dim, num_classes, T, lam):
        super(CSRA, self).__init__()
        self.T = T
        self.lam = lam
        self.head = nn.Conv2d(input_dim, num_classes, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        score = self.head(x) / torch.norm(self.head.weight, dim=1, keepdim=True).transpose(0, 1)
        score = score.flatten(2)
        base_logit = torch.mean(score, dim=2)
        if self.T == 99:
            att_logit = torch.max(score, dim=2)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=2)
        return base_logit + self.lam * att_logit

class MHA(nn.Module):
    temp_settings = {
        1: [1],
        2: [1, 99],
        4: [1, 2, 4, 99],
        6: [1, 2, 3, 4, 5, 99],
        8: [1, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, input_dim, num_classes):
        super(MHA, self).__init__()
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(input_dim, num_classes, self.temp_list[i], lam)
            for i in range(num_heads)
        ])

    def forward(self, x):
        logit = 0.
        for head in self.multi_head:
            logit += head(x)
        return logit

class ResNet_CSRA(ResNet):
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, num_features, num_heads, lam, num_classes, depth=18):
        self.block, self.layers = self.arch_settings[depth]
        self.depth = depth
        super(ResNet_CSRA, self).__init__(self.block, self.layers)
        self.fc1 = nn.Linear(num_features, 64 * 112 * 112)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier = MHA(num_heads, lam, 512, num_classes)
        self.loss_func = F.binary_cross_entropy_with_logits
        self.init_weights(pretrained=True)

    def init_weights(self, pretrained=True, cutmix=None):
        if pretrained:
            model_url = model_urls["resnet{}".format(self.depth)]
            state_dict = model_zoo.load_url(model_url)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc1' not in k and 'classifier' not in k and 'conv1' not in k}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(self, 'fc1'):
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
            if self.fc1.bias is not None:
                nn.init.constant_(self.fc1.bias, 0)
        if hasattr(self, 'classifier'):
            for head in self.classifier.multi_head:
                nn.init.xavier_normal_(head.head.weight)
                if head.head.bias is not None:
                    nn.init.constant_(head.head.bias, 0)

    def forward_train(self, x, target):
        x = self.fc1(x)
        x = x.view(-1, 64, 112, 112)
        x = super().forward(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        logit = self.classifier(x)
        loss = self.loss_func(logit, target, reduction="mean")
        return logit, loss

    def forward_test(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, 112, 112)
        x = super().forward(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward(self, x, target=None):
        if target is not None:
            return self.forward_train(x, target)
        else:
            return self.forward_test(x)
