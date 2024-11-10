from dit import DiT, get_s_model
from diffusion import GaussianDiffusion
import torch
import matplotlib.pyplot as plt
import os

from rectified_flow import RectifiedFlow

def generate_diffusion_ckpt(ckpt_path):
    num_timesteps = 1000
    model, checkpoint = DiT.load(ckpt_path)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    diff = GaussianDiffusion(device, model, num_timesteps=model.cfg.num_timesteps)
    samples = diff.generate((4, 1, 28, 28), y=3, classifier_guidance_strength=4.0)
    os.makedirs("plots", exist_ok=True)
    imgs = samples.cpu().numpy().squeeze()
    
    return imgs

def generate_rf_ckpt(ckpt_path):
    num_timesteps = 20
    model, checkpoint = DiT.load(ckpt_path)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    diff = RectifiedFlow(model, num_timesteps=num_timesteps, device=device)
    samples = diff.generate((8, 1, 28, 28), y=5, classifier_guidance_strength=4.0)
    os.makedirs("plots", exist_ok=True)
    imgs = samples.cpu().numpy().squeeze()
    
    return imgs


if __name__ == "__main__":
    imgs = generate_rf_ckpt("checkpoints/mnist/model-20.pt")
    
    # imgs = generate_diffusion_ckpt("checkpoints/fashing_mnist/model-20.pt")

    for i,img in enumerate(imgs):
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(f"plots/sample_{i}.png", bbox_inches="tight")
        
