import numpy as np
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

def generate_rf_ckpt(ckpt_path, img_size = 32,threed=False):
    num_timesteps = 50
    model, checkpoint = DiT.load(ckpt_path)
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    diff = RectifiedFlow(model, num_timesteps=num_timesteps, device=device)
    all_imgs = []
    for y in range( min(10,model.cfg.num_classes)):
        samples = diff.generate((8, 1 if not threed else 3, img_size, img_size), y=y, classifier_guidance_strength=8.0)
        os.makedirs("plots", exist_ok=True)
        if threed:
            samples = samples.permute(0, 2, 3, 1)
            samples = torch.clamp(samples, -1, 1)
            samples = (samples + 1) / 2
        imgs = samples.cpu().numpy().squeeze()
        all_imgs.append(imgs)
    
    imgs = np.concatenate(all_imgs, axis=0)
    return imgs


if __name__ == "__main__":
    # imgs = generate_rf_ckpt("checkpoints/cifar10/model-20.pt", threed=True)

    # imgs = generate_rf_ckpt("checkpoints/mnist/model-20.pt", threed=False, img_size=28)
    imgs = generate_rf_ckpt("checkpoints/flowers102/model-18.pt", threed=True, img_size=128)
    # imgs = generate_diffusion_ckpt("checkpoints/fashing_mnist/model-20.pt")

    for i,img in enumerate(imgs):
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(f"plots/sample_{i}.png", bbox_inches="tight")
        
