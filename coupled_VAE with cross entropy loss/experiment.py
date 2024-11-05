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

class CVAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(CVAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
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
        # Sample reconstruction and generated images
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

        # Generate and save reconstructions
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(
            recons.data,
            os.path.join(self.logger.log_dir, "Reconstructions", f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=12
        )

        # Generate and save samples
        try:
            samples = self.model.sample(144, self.curr_device, labels=test_label)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=12
            )
        except Warning:
            pass

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