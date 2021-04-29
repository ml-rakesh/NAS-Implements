import torch
import torch.nn as nn

def swish(x, inplace: bool = False):
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)


# Create three blocks
class block1(nn.Module):
    def __init__(self, in_shape):
        super(block1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_shape, in_shape),
            nn.BatchNorm1d(in_shape),
            Swish(),
        )

    def forward(self, x):
        return self.fc1(x)
    
class block2(nn.Module):
    def __init__(self, in_shape):
        super(block2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_shape, in_shape*2),
            nn.BatchNorm1d(in_shape*2),
            Swish(),
            nn.Linear(in_shape*2, in_shape),
            nn.BatchNorm1d(in_shape),
            Swish(),
        )

    def forward(self, x):
        return self.fc1(x)

class block3(nn.Module):
    def __init__(self, in_shape):
        super(block3, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_shape, in_shape*2),
            nn.BatchNorm1d(in_shape*2),
            Swish(),
            nn.Linear(in_shape*2, in_shape*3),
            nn.BatchNorm1d(in_shape*3),
            Swish(),
            nn.Linear(in_shape*3, in_shape*2),
            nn.BatchNorm1d(in_shape*2),
            Swish(),
            nn.Linear(in_shape*2, in_shape),
            nn.BatchNorm1d(in_shape),
            Swish(),
        )
    def forward(self, x):
        return self.fc1(x)
    
class block4(nn.Module):
    def __init__(self, in_shape):
        super(block4, self).__init__()
        self.fc1 = nn.Linear(in_shape, in_shape*2)
        self.bn1 = nn.BatchNorm1d(in_shape*2)
        self.swish = Swish()
        self.fc2 = nn.Linear(in_shape*2, in_shape*4)
        self.bn2 = nn.BatchNorm1d(in_shape*4)
        self.cat_shape = (in_shape*4) + (in_shape)
        self.fc3 = nn.Linear(self.cat_shape, in_shape)
        self.bn3 = nn.BatchNorm1d(in_shape)
    def forward(self, x):
        inp = x
        x1 = self.swish(self.bn1(self.fc1(x)))
        x2 = self.swish(self.bn2(self.fc2(x1)))
        x3 = torch.cat([inp, x2], dim=1)
        x4 = self.swish(self.bn3(self.fc3(x3)))
        return x4

class block5(nn.Module):
    def __init__(self,in_shape):
        super(block5, self).__init__()
        self.fc1 = nn.Linear(in_shape, in_shape*3)
        self.bn1 = nn.BatchNorm1d(in_shape*3)
        self.swish = Swish()
        self.fc2 = nn.Linear(in_shape*3, in_shape)
        self.bn2 = nn.BatchNorm1d(in_shape)
    def forward(self, x):
        inp = x
        x = self.swish(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        #skip connection
        x += inp
        x = self.swish(x)
        return x
    
def build_activation(act_func):
    if act_func == 'relu':
        return nn.ReLU()
    elif act_func == 'relu6':
        return nn.ReLU6()
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'softmax':
        return nn.Softmax(dim=1)
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == 'swish':
        return Swish()
    elif act_func is None:
        return None
    else:
        raise ValueError('do not support: %s' % act_func)