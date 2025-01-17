import os
import yaml
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models import vae_models
from experiment import CVAEXperiment
from dataset import VAEDataset
from callbackLoss import LossLoggerCallback  # Import the LossLoggerCallback


def main():
    parser = argparse.ArgumentParser(description='Coupled VAE model runner')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/coupled_vae.yaml')
    
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return
        
    # Extract parameter values for folder naming
    prior_variance = config['model_params'].get('prior_variance', 1.0)
    kld_weight = config['exp_params'].get('kld_weight', 1.0)
    kappa = config['exp_params'].get('kappa', 0.0)
    n_latent_samples = config['exp_params'].get('n_latent_samples', 1)
    
    # Generate a custom name for the folder based on parameters
    folder_name = (
        f"{config['logging_params']['name']}_"
        f"priorVar_{prior_variance}_"
        f"kldWeight_{kld_weight}_"
        f"kappa_{kappa}_"
        f"nLatentSamples_{n_latent_samples}"
    )
    
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=folder_name)
    
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    
    # Initialize model
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = CVAEXperiment(model, config['exp_params'])
    
    # Load data
    data = VAEDataset(**config["data_params"], pin_memory=False)
    data.setup()
    
    # Ensure directories for saving samples and reconstructions exist
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
    
    # Warm-up forward pass
    print("Performing a warm-up forward pass...")
    warmup_batch = next(iter(data.train_dataloader()))
    warmup_images, _ = warmup_batch  # Get images from the batch
    warmup_images = warmup_images.to('cpu')  # Change 'cpu' to your device if applicable
    experiment.model(warmup_images)  # Run a forward pass to initialize attributes
    
    # Log logvar statistics after warm-up
    if hasattr(model, 'logvar'):
        print(f"Logvar after warm-up: {model.logvar.mean().item()} ± {model.logvar.std().item()}")
    
    # Trainer setup
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True
            ),
            LossLoggerCallback(f"{folder_name}.txt")  # Add the LossLoggerCallback with updated filename
        ],
        accelerator='cpu',  # Change to 'gpu' if you have a GPU and want to use it
        devices=1,  # Change this if you want to use multiple GPUs
        max_epochs=config['trainer_params']['max_epochs'],
    )
    
    # Assign trainer to experiment for sampling
    experiment.trainer = runner
    
    # Generate reconstructions and samples before training starts
    print("Generating reconstructions and samples before training starts...")
    experiment.curr_device = 'cpu'  # Set to your desired device
    experiment.sample_images(data.test_dataloader())
    
    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)
    
    
if __name__ == '__main__':
    main()