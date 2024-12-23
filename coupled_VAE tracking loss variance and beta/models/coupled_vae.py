import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List
from coupled_functions_torch import *


class CoupledVAE(BaseVAE):
    
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(CoupledVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        
        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),  # Change to 3 channels
            nn.Sigmoid()  # Keep Sigmoid for cross-entropy loss
        )
        
    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]
    
    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the loss using regular ELBO (with standard MSE and Gaussian divergence).
    
        :param args: Arguments containing reconstruction, input, mu, and log variance.
        :param kwargs: Keyword arguments, including M_N (KL weight) and kappa.
        :return: A dictionary containing the total loss, reconstruction loss, and KL divergence.
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        # Weight for KL divergence
        kld_weight = kwargs['M_N']
        
        # Compute ELBO using the regular components
        elbo, recons_loss, kld_loss = regular_elbo(recons, input, mu, log_var, kld_weight)
        
        return {
            'loss': elbo,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': -torch.mean(kld_loss).detach()
        }
    
    def sample(self, num_samples: int, current_device: int, prior_variance: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space using the Gaussian prior with a configurable variance.
        
        Args:
            num_samples (int): Number of samples to generate.
            current_device (int): Device to use for tensor computations.
            prior_variance (float): Variance of the Gaussian prior.
            
        Returns:
            samples (torch.Tensor): Decoded samples from the latent space.
        """
        # Sample latent vectors from a Gaussian prior with the specified variance
        std_dev = torch.sqrt(torch.tensor(prior_variance)).to(current_device)
        z = torch.randn(num_samples, self.latent_dim).to(current_device) * std_dev
        samples = self.decode(z)
        return samples
    
    def generate(self, x: torch.Tensor, n_latent_samples: int, n_generated_per_z: int, **kwargs) -> torch.Tensor:
        generated_images = []
        for _ in range(n_latent_samples):
            # Encode and sample latent variable z
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            
            # Generate multiple images for this z
            for _ in range(n_generated_per_z):
                generated_image = self.decode(z)
                generated_images.append(generated_image)
                
        return torch.cat(generated_images, dim=0)




    