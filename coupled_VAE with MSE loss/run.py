import os
import yaml
import argparse
from pathlib import Path
from models import vae_models
from experiment import CVAEXperiment
from dataset import VAEDataset
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

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

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'])

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    # Initialize model
    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = CVAEXperiment(model, config['exp_params'])

    # Load data
    data = VAEDataset(**config["data_params"], pin_memory=False)
    data.setup()

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
        ],
        accelerator='cpu',  # Change to 'gpu' if you have a GPU and want to use it
        devices=1,  # Change this if you want to use multiple GPUs
        max_epochs=config['trainer_params']['max_epochs'],
        #gradient_clip_val=config['trainer_params']['gradient_clip_val'],
    )

    # Ensure directories for saving samples and reconstructions exist
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == '__main__':
    main()