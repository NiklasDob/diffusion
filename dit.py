from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_size: int
    patch_size: int
    in_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    class_dropout_prob: float
    num_classes: int
    num_timesteps: int

def get_s_model(**kwargs):
    return ModelConfig(input_size=28, patch_size=4, in_channels=1, hidden_size=128, depth=6, num_heads=8, class_dropout_prob=0.1, **kwargs)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        v = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(v,v,v)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, num_timesteps=1000):
        super().__init__()
        self.emb = nn.Embedding(num_timesteps, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, t):
        t = self.emb(t)
        return self.mlp(t)
    

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(num_classes, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, y):
        y = self.emb(y)
        return self.mlp(y)

class PatchEmbed(nn.Module):
    def __init__(self, input_size, patch_size, in_channels, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches_per_dim = input_size // patch_size  # Number of patches along one dimension
        self.num_patches = self.num_patches_per_dim ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = nn.LayerNorm(hidden_size)

        # Separate embeddings for row and column positions
        self.row_embed = nn.Embedding(self.num_patches_per_dim, hidden_size)
        self.col_embed = nn.Embedding(self.num_patches_per_dim, hidden_size)

    def forward(self, x):
        # Apply the convolution to extract patch embeddings
        x = self.proj(x)  # (batch_size, hidden_size, num_patches_per_dim, num_patches_per_dim)
        x = x.flatten(2)  # Flatten height and width into a single dimension (batch_size, hidden_size, num_patches)
        x = x.transpose(-1, -2)  # (batch_size, num_patches, hidden_size)

        # Generate row and column indices
        row_indices = torch.arange(self.num_patches_per_dim, device=x.device).unsqueeze(1).repeat(1, self.num_patches_per_dim)
        col_indices = torch.arange(self.num_patches_per_dim, device=x.device).unsqueeze(0).repeat(self.num_patches_per_dim, 1)

        # Apply embeddings for row and column positions
        row_embedding = self.row_embed(row_indices.flatten()).unsqueeze(0)  # (1, num_patches, hidden_size)
        col_embedding = self.col_embed(col_indices.flatten()).unsqueeze(0)  # (1, num_patches, hidden_size)

        # Add positional embeddings to the patch embeddings
        x = x + row_embedding + col_embedding

        # Apply layer normalization
        x = self.norm(x)

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        x = self.unpatchify(x)
        return x

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    


class DiT(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.cfg = model_config

        self.x_embedder = PatchEmbed(self.cfg.input_size, self.cfg.patch_size, self.cfg.in_channels, self.cfg.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.cfg.hidden_size, self.cfg.num_timesteps)
        self.y_embedder = LabelEmbedder(self.cfg.num_classes, self.cfg.hidden_size)

        self.blocks = nn.ModuleList([
            DiTBlock(self.cfg.hidden_size, self.cfg.num_heads) for _ in range(self.cfg.depth)
        ])
        self.final_layer = FinalLayer(self.cfg.hidden_size, self.cfg.patch_size, self.cfg.in_channels)
        self.initialize_weights()

    def save(self, path : str, **kwargs):
        torch.save({"state_dict": self.state_dict(), "model_config" : self.cfg, **kwargs}, path)

    @staticmethod
    def load(path : str):
        checkpoint = torch.load(path, map_location="cpu")
        model = DiT(model_config=checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        del checkpoint["state_dict"]
        return model, checkpoint

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_number_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, t, y = None):
        
        x = self.x_embedder(x)
        t = self.t_embedder(t)
         
        if y is None or (self.training and torch.rand(1).item() < self.cfg.class_dropout_prob):
            c = t
        else:
            y = self.y_embedder(y)
            c = t + y
        
        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)

        return x
    
if __name__ == "__main__":
    num_timesteps = 1000
    num_classes = 1000
    model_cfg = get_s_model(num_timesteps=num_timesteps, num_classes=num_classes)
    model = DiT(model_cfg)
    print(model)
    x = torch.randn((2, 1, 32, 32))
    t = torch.randint(0, num_timesteps, (2,))
    y = torch.randint(0, num_classes, (2,))
    out = model(x, t, y)
    print(out.shape)
    print(model.get_number_parameters())