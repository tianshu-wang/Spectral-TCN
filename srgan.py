import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pixelshuffle1d

def swish(x):
    return x*torch.sigmoid(x)

class ResBlock(nn.Module):
    def __init__(self,nchannel=8,kernel_size=3,stride=1):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv1d(nchannel,nchannel,kernel_size,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm1d(nchannel)
        self.conv2 = nn.Conv1d(nchannel,nchannel,kernel_size,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm1d(nchannel)
    def forward(self,x):
        y = swish(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y))+x

class UpSampling(nn.Module): 
    def __init__(self,nchannel=8,kernel_size=3,stride=1,upfactor=2):
        super(UpSampling,self).__init__()
        self.conv = nn.Conv1d(nchannel,nchannel*upfactor,kernel_size,stride=stride,padding=1)
        self.shuffle = pixelshuffle1d.PixelShuffle1D(upfactor)
    def forward(self,x):
        return self.shuffle(swish(self.conv(x)))

class Generator(nn.Module):
    def __init__(self,n_res,n_up,nchannel=8):
        super(Generator,self).__init__()
        self.n_res = n_res
        self.n_up = n_up
        self.conv1 = nn.Conv1d(1,nchannel,5,stride=1,padding=2)
        for i in range(self.n_res):
            self.add_module('resblock'+str(i+1),ResBlock(nchannel=nchannel))
        self.conv2 = nn.Conv1d(nchannel,nchannel,3,padding=1)
        self.bn2 = nn.BatchNorm1d(nchannel)
        for i in range(self.n_up):
            self.add_module('upsampling'+str(i+1),UpSampling(nchannel=nchannel))
        self.conv3 = nn.Conv1d(nchannel,1,3,padding=1)
    def forward(self,x):
        x = swish(self.conv1(x))
        y = x.clone()
        for i in range(self.n_res):
            y = self.__getattr__('resblock'+str(i+1))(y)
        x = self.bn2(self.conv2(y))+x
        for i in range(self.n_up):
            x = self.__getattr__('upsampling'+str(i+1))(x)
        return F.relu(self.conv3(x))

class Discriminator(nn.Module):
    def __init__(self,nchannel=8):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv1d(1,nchannel,3,padding=1) 

        self.conv2 = nn.Conv1d(nchannel,nchannel,3,stride=2,padding=1) 
        self.bn2 = nn.BatchNorm1d(nchannel)
        self.conv3 = nn.Conv1d(nchannel,2*nchannel,3,stride=1,padding=1) 
        self.bn3 = nn.BatchNorm1d(2*nchannel)
        self.conv4 = nn.Conv1d(2*nchannel,2*nchannel,3,stride=2,padding=1) 
        self.bn4 = nn.BatchNorm1d(2*nchannel)
        self.conv5 = nn.Conv1d(2*nchannel,4*nchannel,3,stride=1,padding=1) 
        self.bn5 = nn.BatchNorm1d(4*nchannel)
        self.conv6 = nn.Conv1d(4*nchannel,4*nchannel,3,stride=2,padding=1) 
        self.bn6 = nn.BatchNorm1d(4*nchannel)
        self.conv7 = nn.Conv1d(4*nchannel,4*nchannel,3,stride=1,padding=1) 
        self.bn7 = nn.BatchNorm1d(4*nchannel)
        self.conv8 = nn.Conv1d(4*nchannel,4*nchannel,3,stride=2,padding=1) 
        self.bn8 = nn.BatchNorm1d(4*nchannel)

        self.conv9 = nn.Conv1d(4*nchannel,1,1,stride=1,padding=1)
    def forward(self,x):
        x = swish(self.conv1(x))
        
        x = swish(self.bn2(self.conv2(x)))
        x = swish(self.bn3(self.conv3(x)))
        x = swish(self.bn4(self.conv4(x)))
        x = swish(self.bn5(self.conv5(x)))
        x = swish(self.bn6(self.conv6(x)))
        x = swish(self.bn7(self.conv7(x)))
        x = swish(self.bn8(self.conv8(x)))

        x = self.conv9(x)
        return torch.sigmoid(F.avg_pool1d(x,x.size()[2])).view(x.size()[0],-1)
