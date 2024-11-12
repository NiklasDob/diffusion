from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torchvision import datasets, transforms
import torchvision
from diffusion import Diffusion
from dit import DiT, ModelConfig, get_s_model, get_m_model
from rectified_flow import RectifiedFlow
from tqdm import tqdm
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingConfig:
    name : str
    transform : torchvision.transforms.Lambda
    num_epochs : int
    batch_size : int
    learning_rate : float


    num_timesteps : int
    num_sample_timesteps : int # Only relevant for rectified flow

    diffusion : Diffusion
    model_cfg : ModelConfig

    get_train_test_dataloaders : callable

# TODO: Combine the multiple datasets using the training config

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])
    cwd = os.path.dirname(__file__)
    data_path = os.path.join(cwd, "data")
    train_dataset = datasets.CIFAR10(data_path, download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = datasets.CIFAR10(data_path, download=True, train=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_timesteps = 1000
    num_classes = 10
    model_cfg = get_s_model(num_timesteps=num_timesteps, num_classes=num_classes, in_channels=3, t_continuous=True, input_size=32)
    model = DiT(model_cfg).to(device)
    start_epoch = 0 
    # model, checkpoint = DiT.load("checkpoints/cifar10/model-7.pt")
    # start_epoch = checkpoint.get("epoch",-1) + 1
    model = model.to(device)
    diffusion = RectifiedFlow(model, num_timesteps=num_timesteps, device=device)
    ckpt_path = os.path.join(cwd, "checkpoints", "cifar10")
    os.makedirs(ckpt_path, exist_ok=True)
    # Train model
    print(model.get_number_parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    old_loss = 100
    for epoch in range(start_epoch,20):
        model.train()
        train_losses = []
        for batchIdx, (img, label) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Train"):
            optimizer.zero_grad()
            
            img, label = img.to(device), label.long().to(device)
            loss = diffusion.train_step(img, y=label)
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            
            if batchIdx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batchIdx+1}, Loss: {loss.item():.4f} Mean Train Loss: {np.mean(train_losses):.4f}")
        
        # Validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for img, label in tqdm(val_loader, total=len(val_loader), desc="Validation"):  # Assuming train_loader is used for validation as well
                img, label = img.to(device), label.long().to(device)
                val_loss = diffusion.train_step(img, y=label)
                val_losses.append(val_loss.item())
        
        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch+1}, Train Loss: {mean_train_loss:.4f}, Validation Loss: {mean_val_loss:.4f}")
        
        if mean_val_loss < old_loss:
            print(f"Saving model new model Train Loss: {mean_train_loss:.4f}")
            old_loss = mean_train_loss
            
            model.save(os.path.join(ckpt_path, f"model-{epoch+1}.pt"), **{"train_loss": old_loss, "epoch": epoch})
            
            samples = diffusion.generate((4, 3, 32, 32), num_sample_timesteps=num_timesteps)
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)
            os.makedirs("plots", exist_ok=True)
            imgs = samples.permute(0, 2, 3, 1).cpu().numpy().squeeze()
            for i, img in enumerate(imgs):
                plt.imshow(img, cmap="gray")
                plt.axis("off")
                plt.savefig(f"plots/sample_{i}.png", bbox_inches="tight")
            

