import torch

# Discriminator Adversarial Loss
def get_disc_loss(real_X , fake_X ,disc_X ,adv_criterion):
    
    real_outputs = disc_X(real_X)
    real_labels = torch.ones_like(real_outputs)
    real_loss = adv_criterion(real_outputs,real_labels)
    
    fake_outputs = disc_X(fake_X.detach())
    fake_labels = torch.ones_like(fake_outputs)
    fake_loss = adv_criterion(fake_outputs,fake_labels)
    
    disc_loss = (real_loss + fake_loss)/2
    
    return disc_loss

# Generator Adversarial Loss
def get_gen_adversarial_loss(real_X , disc_Y , gen_XY , adv_criterion):
    
    fake_Y = gen_XY(real_X)
    fake_outputs = disc_Y(fake_Y)
    
    labels = torch.ones_like(fake_outputs)
    gen_adv_loss = adv_criterion(fake_outputs,labels)
    
    return gen_adv_loss , fake_Y

# Identity Loss
def get_identity_loss(real_X, gen_YX ,identity_criterion ):
    
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X , identity_X)
    
    return identity_loss , identity_X

# Cycle Consistency loss
def get_cycle_loss(real_X , fake_Y, gen_YX , cycle_criterion):
    
    cycle_X = gen_YX(fake_Y.detach())
    cycle_loss = cycle_criterion(real_X,cycle_X)
    
    return cycle_loss , cycle_X

#Get the overall generator loss
def get_gen_loss(real_A, real_B ,gen_AB, gen_BA,
                 disc_A , disc_B,
                 adv_criterion,identity_criterion ,cycle_criterion,
                 lambda_identity = 0.1,lambda_cycle = 10):
    
    gen_adv_loss_AB , fake_B = get_gen_adversarial_loss(real_A,disc_B,gen_AB,adv_criterion)
    gen_adv_loss_BA , fake_A = get_gen_adversarial_loss(real_B,disc_A,gen_BA,adv_criterion)
    gen_adv_loss = gen_adv_loss_AB + gen_adv_loss_BA
    
    identity_loss_AB , identity_A = get_identity_loss(real_A, gen_BA,identity_criterion)
    identity_loss_BA, identity_B = get_identity_loss(real_B, gen_AB,identity_criterion)
    identity_loss = identity_loss_AB + identity_loss_BA
    
    cycle_loss_AB , cycle_A = get_cycle_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_loss_BA , cycle_B = get_cycle_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_loss = cycle_loss_AB + cycle_loss_BA
    
    gen_loss = gen_adv_loss + lambda_identity*identity_loss + lambda_cycle*cycle_loss
    
    return gen_loss, fake_A, fake_B

    