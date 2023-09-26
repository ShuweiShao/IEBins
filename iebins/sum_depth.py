import torch
import torch.nn as nn
import numpy as np

class Sum_depth(nn.Module):
    def __init__(self):
        super(Sum_depth, self).__init__()
        self.sum_conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        sum_k = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        
        sum_k = torch.from_numpy(sum_k).float().view(1, 1, 3, 3)
        self.sum_conv.weight = nn.Parameter(sum_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.sum_conv(x) 
        out = out.contiguous().view(-1, 1, x.size(2), x.size(3))
  
        return out

