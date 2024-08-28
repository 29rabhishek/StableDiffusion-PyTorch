import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.imagenet_dataset import ImageNetDataset
from torch.utils.data import DataLoader, DistributedSampler
from models.unet_base import Unet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

from utils.logger import setup_logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def train(rank, world_size, args):
    ddp_setup(rank, world_size)
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    logger = setup_logger(f"ldm_vae_{rank}_logger", save_dir=train_config['ldm_dir'] , if_train=True, file_name=f"rank_{rank}")
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'imagenet': ImageNetDataset
    }.get(dataset_config['name'])
    device_with_rank = f"cuda:{rank}"
    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=False,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vae_latent_dir_name'])
                                )
    
    # data_loader = DataLoader(im_dataset,
    #                          batch_size=train_config['ldm_batch_size'],
    #                          shuffle=True)
    # for distributed training
    sampler = DistributedSampler(im_dataset)
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=False,  # No need to shuffle when using DistributedSampler
                             sampler=sampler)
    
    # Instantiate the model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config)
    
    # Load VAE ONLY if latents are not to be used or are missing
    if not im_dataset.use_latents:
        print('Loading vae model as latents not present')
        vae = VAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to(device_with_rank)
        vae.eval()
        # Load vae if found
        if os.path.exists(train_config["vae_ckpt_path"]):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(train_config["vae_ckpt_path"],
                                           map_location=device_with_rank))
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    scaler = GradScaler()
    model = DDP(model, device_ids=[rank])

    model.train()
    # Run training
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False


    for epoch_idx in range(num_epochs):
        losses = []
        optimizer.zero_grad()
        for im in data_loader:
            with autocast():
            # im = im.float()
                if not im_dataset.use_latents:
                    with torch.no_grad():
                        _, _, z = vae.encode(im)
            
                # Sample random noise
                noise = torch.randn_like(im).to(device_with_rank)
                
                # Sample timestep
                t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device_with_rank)
                
                # Add noise to images according to timestep
                noisy_im = scheduler.add_noise(im, noise, t)
                noise_pred = model(noisy_im, t)
                
                loss = criterion(noise_pred, noise)
                losses.append(loss.item())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        logger.info(f'GPU[{rank}] epoch: {epoch_idx+1} | Loss: {losses:.4f}')
        if rank == 0:  # Log only on the main process      
            torch.save(model.module.state_dict(), f"{os.path.join(train_config['ldm_dir'],train_config['vae_autoencoder_ckpt_name'])}_{epoch_idx+1}.pth")
            logger.info(f"Checkpoint Saved {train_config['vae_autoencoder_ckpt_name']}_{epoch_idx+1}.pth")
        
    
    destroy_process_group()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/imagenet.yaml', type=str)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs=world_size)
