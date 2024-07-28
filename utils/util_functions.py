import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
from skimage import color
import numpy as np
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

plt.rcParams["figure.figsize"] = (10, 10)

# util function to plot images during training
def show_tensor_images(image_tensor , num_images = 25 , size = (1,28,28)):
    image_tensor = (image_tensor+1)/2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1,*size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def init_weight(m):
    if isinstance(m,nn.Conv2d) or isinstance(m , nn.ConvTranspose2d):
        nn.init.normal_(m.weight,0,0.02)
    if isinstance(m , nn.BatchNorm2d):
        nn.init.normal_(m.weight,0,0.02)
        nn.init.constant_(m.bias,0.0)


class ImageDataset(Dataset):
    def __init__(self,summer_files ,winter_files ,transform = None ,mode = 'train'):
        self.transform = transform
        self.files_A = winter_files
        self.files_B = summer_files
        
        if len(self.files_A) > len(self.files_B):
            self.files_A ,self.files_B = self.files_B ,self.files_A
        self.new_perm()
        assert len(self.files_A)>0
        
    def __len__(self):
        return min(len(self.files_A) ,len(self.files_B))
    
    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]
        
    def __getitem__(self , idx):
        item_A = Image.open(self.files_A[idx% len(self.files_A)])
        item_B = Image.open(self.files_B[self.randperm[idx]])
        
        if self.transform is not None :
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)
        
        if item_A.shape[0] !=3:
            item_A = item_A.repeat(3,1,1)
        if item_B.shape[0] !=3:
            item_B = item_B.repeat(3,1,1)
        if idx == len(self)-1:
            self.new_perm()
        
        return (item_A -0.5)*2 , (item_B -0.5)*2
            