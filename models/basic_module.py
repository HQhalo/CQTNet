import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import time
import os
class BasicModule(torch.nn.Module):
    """
    封装了nn.Module,主要是提供了save和load两个方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))
        self.sub_folder = time.strftime('%m%d_%H:%M:%S')

    def load(self, path, map_location='cuda'):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(torch.load(path, map_location=torch.device(map_location)))

    def save(self, epoch, songMrr, prefix):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        save_folder = prefix + '/'+self.sub_folder
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        name = save_folder + '/' + str(epoch) + '_' + str(round(songMrr, 3)) + '.pth'
        print('Save model to', name)
        torch.save(self.state_dict(), name)
        # torch.save(self.state_dict(), prefix+'/latest.pth')
        return name
    
    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def load_latest(self, notes):
        path = 'check_points/' + self.model_name +notes+ '/latest.pth'
        self.load_state_dict(torch.load(path))