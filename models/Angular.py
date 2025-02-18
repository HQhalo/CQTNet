
from torch import nn
import torch.nn.functional as F
from .basic_module import BasicModule
import torch
from .CQTNet import CQTNet
from .TPPNet import CQTTPPNet

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
        
class CQTNetAngular(BasicModule):
    def __init__(self, num_classes=4611, loss_type='cosface'):
        super(CQTNetAngular, self).__init__()
        self.cqtnet = CQTNet()

        self.adms_loss = AngularPenaltySMLoss(300, num_classes, loss_type=loss_type)

    def forward(self, x, labels=None, embed=True):
        x, feature = self.cqtnet(x)
        if embed:
            return x, feature
        L = self.adms_loss(feature, labels)
        return L

class CQTTPPNetAngular(BasicModule):
    def __init__(self, num_classes=5000, loss_type='cosface'):
        super(CQTTPPNetAngular, self).__init__()
        self.cqtnet = CQTTPPNet()

        self.adms_loss = AngularPenaltySMLoss(300, num_classes, loss_type=loss_type)

    def forward(self, x, labels=None, embed=True):
        x, feature = self.cqtnet(x)
        if embed:
            return x, feature
        L = self.adms_loss(feature, labels)
        return L