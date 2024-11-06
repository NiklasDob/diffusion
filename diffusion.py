import torch
from tqdm import tqdm
import torch.nn.functional as F

class GaussianDiffusion:
    def __init__(self, device, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.2, clip_sample=True):
        self.device = device
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.clip_sample = clip_sample
        self.final_alpha_cumprod = torch.tensor(1.0).to(device)

        # Define the beta schedule and calculate alpha values
        self.beta = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = torch.ones_like(self.alpha_cumprod)
        self.alpha_cumprod_prev[1:] = self.alpha_cumprod_prev[:-1].clone()

    @torch.no_grad()
    def q(self, x_0, t):
        """
        Adds noise to the image at a given timestep t.
        x_0: the original image
        t: the timestep for noise level
        Returns the noisy image at timestep t
        """
        alpha_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0).to(self.device)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return x_t, noise

    @torch.no_grad()
    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.num_timesteps, (batch_size,), device=self.device, dtype=torch.long)

    def train_step(self, x_0,y = None):
        """
        Performs a single training step.
        """
        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size)

        # Forward diffusion: Add noise to input image
        x_t, noise = self.q(x_0, t)

        # Predict the noise using the model
        predicted_noise = self.model(x_t, t, y=y)

        # Loss: Mean Squared Error between predicted and actual noise
        loss = F.mse_loss(predicted_noise, noise)
        return loss


    @torch.no_grad()
    def generate(self, shape, y=None, eta=1.0, generator=None, classifier_guidance_strength=1.0):
        """
        Samples a new image by running the reverse process.
        shape: the shape of the image to generate
        Returns the generated image
        """
        x_t = torch.randn(shape).to(self.device)  # Start with pure noise

        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            ts = torch.ones(shape[0], device=self.device) * t
            if y is not None:
                y_val = torch.ones_like(ts) * y
                y_val = y_val.long()
            else:
                y_val = None
            ts = ts.long()
            if y is not None:
                eps_y = self.model(x_t, ts, y=y_val)
                eps_no_cond = self.model(x_t, ts, y=None)
                predicted_noise = (1 + classifier_guidance_strength) * eps_y - classifier_guidance_strength * eps_no_cond
            else:
                predicted_noise = self.model(x_t, ts, y=y_val)

            alpha_t_c = self.alpha_cumprod[t].view(-1, 1, 1, 1)
            alpha_t = self.alpha[ts].view(-1, 1, 1, 1)

            # if t>1:
            #     z = torch.randn_like(x_t)
            # else:
            #     z = torch.zeros_like(x_t)

            # x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((1-alpha_t) / torch.sqrt(1-alpha_t_cum)) * predicted_noise)  + torch.sqrt(var) *  z
            if t == 1:
                z = torch.zeros_like(x_t)  
            else:
                z = torch.randn_like(x_t)

            # x = 1/np.sqrt(alphas[t])*(x - ((betas[t]) / np.sqrt(1-alphas_prod[t]))*eps) + betas[t]*z
            beta_t = self.beta[ts].view(-1, 1, 1, 1)
            sigma_t = eta * torch.sqrt(beta_t * (1 - self.alpha_cumprod_prev[ts].view(-1, 1, 1, 1)) / (1 - self.alpha_cumprod[ts].view(-1, 1, 1, 1)))

            # x_t = (1 / torch.sqrt(alpha_t)) * (x_t - ((beta_t) / torch.sqrt(1-alpha_t_c)) * predicted_noise)  + torch.sqrt(beta_t) *  z
            x_t = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_t_c)) * predicted_noise) + sigma_t * z

            # print(x_t)
            x_t = torch.clamp(x_t, -1, 1)

        # Rescale x_t to [0, 1] range if normalized output is desired
        x_t = torch.clamp(x_t, -1, 1)
        # x_t = (x_t + 1) / 2
    

        return x_t