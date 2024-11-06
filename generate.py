from dit import DiT, get_s_model
from diffusion import GaussianDiffusion
import torch
import matplotlib.pyplot as plt
import os
if __name__ == "__main__":

    num_timesteps = 1000
    model, checkpoint = DiT.load("checkpoints/model-15.pt")
    print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    diff = GaussianDiffusion(device, model, num_timesteps=model.cfg.num_timesteps)
    samples = diff.generate((4, 1, 28, 28), y=None)
    os.makedirs("plots", exist_ok=True)
    imgs = samples.cpu().numpy().squeeze()
    for i,img in enumerate(imgs):
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.savefig(f"plots/sample_{i}.png", bbox_inches="tight")
        
