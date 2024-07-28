# Importing the necessary libraries
import torch
from torch import nn 

# definition of the contracting block
class ConvolutionalLayer(nn.Module):
    def __init__(self, in_channels , use_bn = True, kernel_size = 3,activation = 'relu'):
        super(ConvolutionalLayer,self).__init__()
        self.conv = nn.Conv2d(in_channels ,in_channels*2 ,padding=1 ,stride =2 ,kernel_size = kernel_size ,padding_mode = 'reflect')
        self.activation = nn.ReLU() if activation=='relu'else nn.LeakyReLU(0.2)
        if use_bn:
            self.instance_norm = nn.InstanceNorm2d(in_channels*2)
        self.use_bn = use_bn
        
    def forward(self,obs):
        x = self.conv(obs)
        if self.use_bn :
            x = self.instance_norm(x)
        x = self.activation(x)
        return x
    

# definition of the residual block
class ResidualBlock(nn.Module):
    def __init__(self , in_channels):
        super(ResidualBlock ,self).__init__()
        self.conv1 = nn.Conv2d(in_channels ,in_channels ,padding=1  ,kernel_size = 3 ,padding_mode = 'reflect')
        self.conv2 = nn.Conv2d(in_channels ,in_channels ,padding=1  ,kernel_size = 3 ,padding_mode = 'reflect')
        self.instance_norm = nn.InstanceNorm2d(in_channels)
        self.activation = nn.ReLU()
    
    def forward(self ,obs):
        original_x = obs.clone()
        x = self.conv1(obs)
        x = self.instance_norm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instance_norm(x)
        return original_x + x
    
# definition of the expanding block
class TransposeConvolutionalLayer(nn.Module):
    def __init__(self, in_channels,use_bn=True):
        super(TransposeConvolutionalLayer , self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels,in_channels//2 ,kernel_size = 3 , stride = 2 , padding =1 , output_padding = 1)
        if use_bn:
            self.instance_norm = nn.InstanceNorm2d(in_channels*2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        
    def forward(self , obs):
        x = self.upconv(obs)
        if self.use_bn:
            x = self.instance_norm(x)
        x = self.activation(x)
        return x
    

# Definition of the feature map layer of the U-net
class FeatureMapBlock(nn.Module):
    def __init__(self , in_channels , out_channels):
        super(FeatureMapBlock , self).__init__()
        self.conv = nn.Conv2d(in_channels ,out_channels , kernel_size = 7, padding = 3, padding_mode = 'reflect')
        
    def forward(self , obs):
        x = self.conv(obs)
        return x
    

# Putting it all together : building the generator
class Generator(nn.Module):
    def __init__(self,in_channels , out_channels, hidden_channels= 64):
        super(Generator , self).__init__()
        self.upfeature = FeatureMapBlock(in_channels ,hidden_channels)
        self.contract1 = ConvolutionalLayer(hidden_channels)
        self.contract2 = ConvolutionalLayer(hidden_channels*2)
        res_mult = 4
        self.res0 = ResidualBlock(hidden_channels*res_mult)
        self.res1 = ResidualBlock(hidden_channels*res_mult)
        self.res2 = ResidualBlock(hidden_channels*res_mult)
        self.res3 = ResidualBlock(hidden_channels*res_mult)
        self.res4 = ResidualBlock(hidden_channels*res_mult)
        self.res5 = ResidualBlock(hidden_channels*res_mult)
        self.res6 = ResidualBlock(hidden_channels*res_mult)
        self.res7 = ResidualBlock(hidden_channels*res_mult)
        self.res8 = ResidualBlock(hidden_channels*res_mult)
        self.expand1 = TransposeConvolutionalLayer(hidden_channels*4)
        self.expand2 = TransposeConvolutionalLayer(hidden_channels*2)
        self.downfeature = FeatureMapBlock(hidden_channels ,out_channels)
        self.tanh = nn.Tanh()
    
    def forward(self,obs):
        x0 = self.upfeature(obs)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.res0(x2)
        x4 = self.res1(x3)
        x5 = self.res2(x4)
        x6 = self.res3(x5)
        x7 = self.res4(x6)
        x8 = self.res5(x7)
        x9 = self.res6(x8)
        x10 = self.res7(x9)
        x11 = self.res8(x10)
        x12 = self.expand1(x11)
        x13 = self.expand2(x12)
        xn = self.downfeature(x13)
        
        return self.tanh(xn)
