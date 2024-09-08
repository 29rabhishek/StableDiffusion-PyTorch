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
from torch.utils.data import DataLoader
from models.unet_base import Unet
from models.vae import VAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from accelerate import Accelerator
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):

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

    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'imagenet': ImageNetDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'],
                                use_latents=False,
                                latent_path=os.path.join(train_config['task_name'],
                                                         train_config['vae_latent_dir_name'])
                                )
    
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)
    
    # Instantiate the model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to('cuda')
    
    # Load VAE ONLY if latents are not to be used or are missing
    if not im_dataset.use_latents:
        print('Loading vae model as latents not present')
        vae = VAE(im_channels=dataset_config['im_channels'],
                    model_config=autoencoder_model_config).to('cuda')
        vae.eval()
        # Load vae if found
        if os.path.exists(train_config["vae_ckpt_path"]):
            print('Loaded vae checkpoint')
            vae.load_state_dict(torch.load(train_config["vae_ckpt_path"],
                                           map_location='cuda'))
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()



    model.train()
    # Run training
    if not im_dataset.use_latents:
        for param in vae.parameters():
            param.requires_grad = False


    for epoch_idx in range(num_epochs):
        losses = []
        for im in data_loader:
            optimizer.zero_grad()
            im = im.to('cuda').float()
            if not im_dataset.use_latents:
                with torch.no_grad():
                    _, _, z = vae.encode(im)
            
            # Sample random noise
            noise = torch.randn_like(im).to('cuda')
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to('cuda')
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)
            
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
        print(f" Epoch: {epoch_idx+1}, loss: {loss}")

    #     if epoch_idx % diffusion_config["ckpt_save_epoch"] == 0:
    #         #saving checkpoint
    #         accelerator.wait_for_everyone()
    #         if accelerator.is_main_process:
    #             unwrapped_model = accelerator.unwrap_model(model)
    #             accelerator.save(unwrapped_model.state_dict(), os.path.join(train_config['task_name'], f"{train_config['ldm_ckpt_name']}"))
        
    # # accelerator.print('Done Training ...')


    # # accelerator.end_training()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/imagenet.yaml', type=str)
    args = parser.parse_args()
    train(args)
