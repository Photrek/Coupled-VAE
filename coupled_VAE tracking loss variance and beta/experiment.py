import os
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from coupled_functions_torch import *

class CVAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(CVAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params  # Already includes all parameters
        self.n_latent_samples = params.get('n_latent_samples') 
        self.n_generated_per_z = params.get('n_generated_per_z') 
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        self.automatic_optimization = False  # Disable automatic optimization
            
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device
        
        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params['kld_weight'],  # Scaling factor for KL divergence
            kappa=self.params.get('kappa'),
            batch_idx=batch_idx
        )
        
        # Log training losses
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        
        # Manual optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(train_loss['loss'])
        
        # Manually clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
        
        opt.step()
        
        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # Scaling factor for KL divergence during validation
            batch_idx=batch_idx
        )

        # Log validation losses
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get a batch of test images
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input[:60].to(self.curr_device)
        test_label = test_label[:60].to(self.curr_device)
        
        # Generate and save the reconstructions of original test images
        recons = self.model.generate(
            test_input,
            n_latent_samples=self.n_latent_samples,
            n_generated_per_z=self.n_generated_per_z,
            labels=test_label
        )
        
        # Save reconstructed images
        vutils.save_image(
            recons.data,
            os.path.join(self.logger.log_dir, "Reconstructions", f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=10
        )
        
        # Optional: Random sampling using the updated prior variance
        try:
            prior_variance = self.params.get('prior_variance', 1.0)
            samples = self.model.sample(60, self.curr_device, prior_variance=prior_variance)
            
            vutils.save_image(
                samples.cpu().data,
                os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_random_samples_Epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=10
            )
        except Warning as e:
            print(f"Warning during random sampling: {e}")
        
#   def sample_images(self):
#       # Get a batch of test images
#       test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
#       test_input = test_input[:60].to(self.curr_device)
#       test_label = test_label[:60].to(self.curr_device)
#       
#       # Generate and save the reconstructions of original test images
#       recons = self.model.generate(
#           test_input,
#           n_latent_samples=self.n_latent_samples,
#           n_generated_per_z=self.n_generated_per_z,
#           labels=test_label
#       )
#       
#       if self.current_epoch == 0:
#           # Save the original images at epoch 0
#           vutils.save_image(
#               test_input.data,
#               os.path.join(self.logger.log_dir, "Reconstructions", "original_images_epoch_0.png"),
#               normalize=True,
#               nrow=10
#           )
#           
#       # Save reconstructed images
#       vutils.save_image(
#           recons.data,
#           os.path.join(self.logger.log_dir, "Reconstructions", f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
#           normalize=True,
#           nrow=10
#       )
#       
#       # Coupled sampling for generated images
#       all_generated_images = []
#       for idx in range(test_input.size(0)):  # Iterate over each input
#           input_image = test_input[idx].unsqueeze(0)  # Single image batch
#       
#           mu, log_var = self.model.encode(input_image)
#       
#           # Sample multiple `z` values directly using reparameterization
#           std_dev = torch.exp(0.5 * log_var)
#           normal_samples = torch.randn(self.n_latent_samples, mu.size(-1), device=self.curr_device)  # [n_latent_samples, latent_dim]
#           z_samples = mu + std_dev * normal_samples  # [n_latent_samples, latent_dim]
#       
#           # Compute coupled probabilities
#           q_values = torch.exp(-0.5 * ((z_samples - mu) ** 2) / (std_dev ** 2)) / (std_dev * torch.sqrt(torch.tensor(2 * torch.pi)))
#           
#           coupled_probs = compute_coupled_probability(q_values, self.params['alpha'], self.params['kappa'], mu.size(-1))  # [n_latent_samples, latent_dim]
#       
#           # Select `z` values based on coupled probabilities
#           selected_z_indices = torch.multinomial(coupled_probs.mean(dim=-1), num_samples=self.n_generated_per_z, replacement=True)
#           selected_z_samples = z_samples[selected_z_indices]  # [n_generated_per_z, latent_dim]
#       
#           # Decode each selected `z`
#           for z_idx, z in enumerate(selected_z_samples):
#               generated_image = self.model.decode(z.unsqueeze(0))  # Decode one at a time
#               all_generated_images.append(generated_image)
#               
#       # Combine and save all generated images
#       all_generated_images = torch.cat(all_generated_images, dim=0)  # [total_images, channels, height, width]
#       
#       # Calculate the number of rows dynamically for a square layout
#       num_images = all_generated_images.size(0)
#       grid_size = int(num_images ** 0.5)  # Approximate square root
#       vutils.save_image(
#           all_generated_images.data,
#           os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_CoupledSamples_Epoch_{self.current_epoch}.png"),
#           normalize=True,
#           nrow=grid_size  # Use the calculated grid size
#       )
#       
#       # Optional: Random sampling
#       try:
#           samples = self.model.sample(60, self.curr_device, labels=test_label)
#           
#           vutils.save_image(
#               samples.cpu().data,
#               os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_random_samples_Epoch_{self.current_epoch}.png"),
#               normalize=True,
#               nrow=10
#           )
#       except Warning as e:
#           print(f"Warning during random sampling: {e}")
            
    def configure_optimizers(self):
        optims = []
        scheds = []

        # Primary optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        # Check for a secondary optimizer
        if 'LR_2' in self.params and self.params['LR_2'] is not None:
            optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(), lr=self.params['LR_2'])
            optims.append(optimizer2)

        # Primary scheduler
        if 'scheduler_gamma' in self.params and self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optims[0], gamma=self.params['scheduler_gamma'])
            scheds.append(scheduler)

            # Secondary scheduler if present
            if len(optims) > 1 and 'scheduler_gamma_2' in self.params and self.params['scheduler_gamma_2'] is not None:
                scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1], gamma=self.params['scheduler_gamma_2'])
                scheds.append(scheduler2)

        return optims, scheds if scheds else optims