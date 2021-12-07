import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import math
from .basic_module import BasicModule

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features
        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
        
class CQTNetAngular(nn.Module):
    def __init__(self, num_classes=900, loss_type='arcface'):
        super(CQTNetAngular, self).__init__()
        self.cqtnet = CQTNet()

        self.adms_loss = AngularPenaltySMLoss(300, num_classes, loss_type=loss_type)

    def forward(self, x, labels=None, embed=True):
        x, feature = self.cqtnet(x)
        if embed:
            return x, feature
        L = self.adms_loss(feature, labels)
        return L
class CQTNet(BasicModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 32, kernel_size=(12, 3), dilation=(1, 1), padding=(6, 0), bias=False)),
            ('norm0', nn.BatchNorm2d(32)), ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(32, 64, kernel_size=(13, 3), dilation=(1, 2), bias=False)),
            ('norm1', nn.BatchNorm2d(64)), ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv2', nn.Conv2d(64, 64, kernel_size=(13, 3), dilation=(1, 1), bias=False)),
            ('norm2', nn.BatchNorm2d(64)), ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 64, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm3', nn.BatchNorm2d(64)), ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv4', nn.Conv2d(64, 128, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm4', nn.BatchNorm2d(128)), ('relu4', nn.ReLU(inplace=True)),
            ('conv5', nn.Conv2d(128, 128, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm5', nn.BatchNorm2d(128)), ('relu5', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv6', nn.Conv2d(128, 256, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm6', nn.BatchNorm2d(256)), ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(256, 256, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm7', nn.BatchNorm2d(256)), ('relu7', nn.ReLU(inplace=True)),
            ('pool7', nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 1))),

            ('conv8', nn.Conv2d(256, 512, kernel_size=(3, 3), dilation=(1, 1), bias=False)),
            ('norm8', nn.BatchNorm2d(512)), ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(512, 512, kernel_size=(3, 3), dilation=(1, 2), bias=False)),
            ('norm9', nn.BatchNorm2d(512)), ('relu9', nn.ReLU(inplace=True)),
        ]))
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc0 = nn.Linear(512, 300)
        self.fc1 = nn.Linear(300, 10000)

    def forward(self, x):
        # input [N, C, H, W] (W = 396)
        N = x.size()[0]
        x = self.features(x)  # [N, 512, 57, 2~15]
        x = self.pool(x)
        x = x.view(N, -1)
        feature = self.fc0(x)
        x = self.fc1(feature)
        return x, feature
