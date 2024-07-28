import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.generator import Generator
from utils.discriminator import Discriminator
from utils.losses import get_disc_loss , get_gen_loss
from utils.util_functions import show_tensor_images, ImageDataset ,init_weight

from glob import glob
from tqdm.auto import tqdm

#============== getting the images list================================================
summer_images = glob('data/trainA/*')
winter_images = glob('data/trainB/*')


#=================== Hyperparameters definition ======================
adv_criterion = nn.MSELoss()
recon_criterion = nn.L1Loss() #the l1 loss for cycle consistency loss and identity loss 

n_epochs = 100
dim_A = 3 # summer image channels
dim_B = 3 # winter image channels

display_step = 200 #display every 200 steps
batch_size = 1
lr = 0.0002

load_shape = 286
target_shape = 256
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# =================== Dataset definition ==================================
# defining the transformation to apply to images
transform = transforms.Compose([transforms.Resize(load_shape),
                               transforms.RandomCrop(target_shape),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])

#instantiating the dataset
dataset = ImageDataset(summer_files = summer_images ,winter_files = winter_images,transform = transform ,mode = 'train')


# ============== Networks initializations ===================================
# Initialisons les générateurs et les discriminateurs
gen_AB = Generator(dim_A , dim_B).to(device)
gen_BA = Generator(dim_B , dim_A).to(device)
gen_optim = torch.optim.Adam( list(gen_AB.parameters()) + list(gen_BA.parameters()) , lr = lr , betas =(0.5,0.999))

disc_A = Discriminator(dim_A).to(device)
discA_optim = torch.optim.Adam(disc_A.parameters() , lr= lr , betas = (0.5,0.999))

disc_B = Discriminator(dim_B).to(device)
discB_optim = torch.optim.Adam(disc_B.parameters() , lr= lr , betas = (0.5,0.999))

# Initialisons nos NNs
is_pretrained = False

if is_pretrained:
    pre_dict = torch.load('model_checkpoint/cycleGAN_21.pth')
    gen_AB.load_state_dict(pre_dict['gen_AB'])
    gen_BA.load_state_dict(pre_dict['gen_BA'])
    gen_optim.load_state_dict(pre_dict['gen_opt'])
    disc_A.load_state_dict(pre_dict['disc_A'])
    discA_optim.load_state_dict(pre_dict['disc_A_opt'])
    disc_B.load_state_dict(pre_dict['disc_B'])
    discB_optim.load_state_dict(pre_dict['disc_B_opt'])
else:
    gen_AB = gen_AB.apply(init_weight)
    gen_BA = gen_BA.apply(init_weight)
    disc_A = disc_A.apply(init_weight)
    disc_B = disc_B.apply(init_weight)


# Training Function

def train(save_model=False):
    mean_generator_loss = 0.0
    mean_discriminator_loss = 0.0
    dataloader = DataLoader(dataset , batch_size = batch_size , shuffle = True)
    cur_step = 0
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(n_epochs):
        for real_A , real_B in tqdm(dataloader):
            
            # input processing
            real_A = nn.functional.interpolate(real_A , size = target_shape)
            real_B = nn.functional.interpolate(real_B , size = target_shape)
            cur_batch_size = len(real_A)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            #Update Discriminator A
            discA_optim.zero_grad()
            with torch.no_grad():
                fake_A = gen_BA(real_B)
            discA_loss = get_disc_loss(real_A , fake_A ,disc_A ,adv_criterion)
            discA_loss.backward(retain_graph = True)
            discA_optim.step()
            
            # Update Discriminator B
            discB_optim.zero_grad()
            with torch.no_grad():
                fake_B = gen_AB(real_A)
            discB_loss = get_disc_loss(real_B , fake_B ,disc_B ,adv_criterion)
            discB_loss.backward(retain_graph = True)
            discB_optim.step()
            
            # Update the generator
            gen_optim.zero_grad()
            gen_loss, fake_A, fake_B = get_gen_loss(real_A, real_B ,gen_AB, gen_BA,
                                                    disc_A , disc_B,
                                                    adv_criterion,recon_criterion ,recon_criterion)
            gen_loss.backward()
            gen_optim.step()
            
            #Keep track of the discriminator loss
            mean_discriminator_loss += discA_loss.item() / display_step
            
            #Keep track of the generator loss
            mean_generator_loss += gen_loss.item()/ display_step
            
            
            #Visualization and Saving Code
            if cur_step%display_step == 0 :
                
                print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                show_tensor_images(torch.cat([real_A, real_B]), size=(dim_A, target_shape, target_shape))
                show_tensor_images(torch.cat([fake_B, fake_A]), size=(dim_B, target_shape, target_shape))
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if save_model:
                    torch.save({
                        'gen_AB': gen_AB.state_dict(),
                        'gen_BA': gen_BA.state_dict(),
                        'gen_opt': gen_optim.state_dict(),
                        'disc_A': disc_A.state_dict(),
                        'disc_A_opt': discA_optim.state_dict(),
                        'disc_B': disc_B.state_dict(),
                        'disc_B_opt': discB_optim.state_dict()
                    }, f"model_checkpoint/cycleGAN_{epoch}.pth")

            cur_step +=1


if __name__ == '__main__':
    train(save_model=True)