import torch
from torch import nn 
from .generator import FeatureMapBlock , ConvolutionalLayer

# definition of the PatchGan generator
class Discriminator(nn.Module): 
    def __init__(self , in_channels , hidden_channels=64):
        super(Discriminator , self).__init__()
        self.upfeature = FeatureMapBlock(in_channels,hidden_channels)
        self.contract1 = ConvolutionalLayer(hidden_channels , kernel_size = 4 , use_bn = False ,activation = 'lrelu')
        self.contract2 = ConvolutionalLayer(hidden_channels*2 , kernel_size = 4 ,activation = 'lrelu')
        self.contract3 = ConvolutionalLayer(hidden_channels*4 , kernel_size = 4 ,activation = 'lrelu')
        self.final = nn.Conv2d(hidden_channels*8 ,1,kernel_size=1)
        
    def forward(self,obs):
        x0 = self.upfeature(obs)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        return self.final(x3)