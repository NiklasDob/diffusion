from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from dit import DiT, ModelConfig, get_s_model, get_m_model
from rectified_flow import RectifiedFlow
from tqdm import tqdm
import os

def generate_and_plot_samples(img_size, num_timesteps, in_channels, diffusion):
    samples = diffusion.generate((4, in_channels, img_size, img_size), num_sample_timesteps=25)
    # min_val = torch.min(samples.view((samples.shape[0],-1)), dim=1)[0]
    # max_val = torch.max(samples.view((samples.shape[0],-1)), dim=1)[0]
    # samples = (samples - min_val.view((samples.shape[0],1,1,1))) / (max_val - min_val).view((samples.shape[0],1,1,1))
    
    samples = samples.permute(0, 2, 3, 1)
    samples = torch.clamp(samples, -1, 1)
    samples = (samples + 1) / 2
    os.makedirs("plots", exist_ok=True)
    imgs = samples.cpu().numpy().squeeze()
    for i, img in enumerate(imgs):
        plt.imshow(img)
        plt.axis("off")
        plt.savefig(f"plots/sample_{i}.png", bbox_inches="tight")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


img_size = 128


transforms = transforms.Compose([
	transforms.Resize(256),
    transforms.RandomCrop(img_size),
	transforms.ToTensor(),
	transforms.Lambda(lambda x: x * 2 - 1)
])
cwd = os.path.dirname(__file__)
data_path = os.path.join(cwd, "data")
# Dont know why but test is much larger than train?
train_dataset = datasets.Flowers102(data_path, download=True, split="test", transform=transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = datasets.Flowers102(data_path, download=True, split="train", transform=transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

num_timesteps = 1000
num_classes = 102 
in_channels = 3
model_cfg =  ModelConfig(input_size=img_size, patch_size=4, in_channels=in_channels, hidden_size=128, depth=8, num_heads=8, class_dropout_prob=0.1,  t_continuous=True, num_classes=num_classes, num_timesteps=num_timesteps) 
model = DiT(model_cfg).to(device)
start_epoch = 0 
# model, checkpoint = DiT.load("checkpoints/flowers102/model-12.pt")
# start_epoch = checkpoint.get("epoch",-1) + 1
model = model.to(device)
diffusion = RectifiedFlow(model, num_timesteps=num_timesteps, device=device)

generate_and_plot_samples(img_size, num_timesteps, in_channels, diffusion)

ckpt_path = os.path.join(cwd, "checkpoints", "flowers102")
os.makedirs(ckpt_path, exist_ok=True)
# Train model
print(model.get_number_parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
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
        
        generate_and_plot_samples(img_size, num_timesteps, in_channels, diffusion)
        

