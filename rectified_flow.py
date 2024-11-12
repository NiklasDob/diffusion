# https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from diffusion import Diffusion

class RectifiedFlow(Diffusion):
    def __init__(self, model : nn.Module , num_timesteps = 1000, device = "cpu"):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

    def noisify(self, z1):
        z0 = torch.randn_like(z1)

        # dt = 1 / self.num_timesteps
        t = torch.rand((z1.shape[0], 1), device=self.device) #* (1 - dt) + dt 
        t_ = t.view(-1, 1, 1, 1)
        z_t =  t_ * z1 + (1. - t_) * z0  
        target = z1 - z0
        return z_t, t, target
    
    
    def train_step(self, x_0,y = None):
        """
        Performs a single training step.
        """

        x_t, t, flow = self.noisify(x_0)

        predicted_noise = self.model(x_t, t, y=y)

        loss = F.mse_loss(predicted_noise, flow)
        return loss
        
    @torch.no_grad()
    def generate(self, input_shape = [4, 3, 32, 32], num_sample_timesteps : int = None, y : int=None, classifier_guidance_strength : float=4.0):
        if num_sample_timesteps is not None:
            N = num_sample_timesteps
        else:
            N = self.num_timesteps

        num_samples = input_shape[0]   
        z = torch.randn(input_shape, device=self.device)
        if y is not None:
            y_val = torch.ones((num_samples,), device=self.device) * y
            y_val = y_val.long()
        else:
            y_val = None
        dt = 1/N
        for delta_time in tqdm(torch.linspace(0.0, 1.0, N, device=self.device), desc="Sampling RF", total=N):
            t = torch.ones((num_samples,1),device=self.device) * delta_time
            if y is not None:
                eps_y = self.model(z, t, y=y_val)
                eps_no_cond = self.model(z, t, y=None)
                pred = (1 + classifier_guidance_strength) * eps_y - classifier_guidance_strength * eps_no_cond
            else:
                pred = self.model(z, t, y=y_val)

            # t = t.view(-1, 1, 1, 1)
            # z = (1 - t) * pred +  t* z

            # t = torch.ones((num_samples,1),device=self.device) * dt
            # t = t.view(-1, 1, 1, 1)
            # z = (1 - t) * pred +  t* z
            z = z + dt*pred

        return z

