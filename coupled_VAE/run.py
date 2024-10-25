import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import CVAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset  


def main():
    parser = argparse.ArgumentParser(description='Coupled VAE model runner')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/coupled_vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return

    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'],)

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    experiment = CVAEXperiment(model, config['exp_params'])

    
    data = VAEDataset(
        **config["data_params"],
        pin_memory=False 
    )

    data.setup()

    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=2,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True
            ),
        ],
        accelerator='cpu',  
        devices=1,           
        max_epochs= 1,  
        #strategy='ddp'
    )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, datamodule=data)

if __name__ == '__main__':
    main()
    