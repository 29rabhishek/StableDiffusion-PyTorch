import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from models.vae import VAE
from torch.utils.data import DataLoader, DistributedSampler
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
from utils.logger import setup_logger
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

def kl_divergence_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def train(rank, args):
    # Initialize process group for distributed training
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(rank)
    
    logger = setup_logger("VAE_logger", if_train=True)

    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(rank)
    model = DDP(model, device_ids=[rank])

    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])

    sampler = DistributedSampler(im_dataset)
    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=False,  # No need to shuffle when using DistributedSampler
                             sampler=sampler)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    num_epochs = train_config['autoencoder_epochs']
    recon_criterion = torch.nn.MSELoss()

    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    scaler = GradScaler()
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    step_count = 0

    model.train()
    for epoch_idx in range(num_epochs):
        recon_losses = []
        kl_losses = []
        losses = []

        optimizer_g.zero_grad()

        for im in tqdm(data_loader, desc=f"Epoch {epoch_idx+1}/{num_epochs}"):
            step_count += 1
            im = im.float().to(rank)

            with autocast():  # Mixed precision training
                model_output = model(im)
                output, mu, logvar, quantize_losses = model_output

                if step_count % image_save_steps == 0 or step_count == 1:
                    if rank == 0:  # Save images only on the main process
                        sample_size = min(8, im.shape[0])
                        save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
                        save_output = ((save_output + 1) / 2)
                        save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

                        grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
                        img = torchvision.transforms.ToPILImage()(grid)
                        if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                            os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                        img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                              'current_autoencoder_sample_{}.png'.format(img_save_count)))
                        img_save_count += 1
                        img.close()

                recon_loss = recon_criterion(output, im)
                recon_losses.append(recon_loss.item())
                recon_loss = recon_loss / acc_steps

                kl_loss = kl_divergence_loss(mu, logvar)
                kl_losses.append(kl_loss.item())
                kl_loss = kl_loss / acc_steps

                g_loss = recon_loss + kl_loss
                losses.append(g_loss.item())

            scaler.scale(g_loss).backward()

            if step_count % acc_steps == 0:
                scaler.step(optimizer_g)
                scaler.update()
                optimizer_g.zero_grad()

        if rank == 0:  # Log only on the main process
            logger.info(f'Epoch {epoch_idx+1}/{num_epochs} | '
                        f'Total Loss: {np.mean(losses)} | '
                        f'Recon Loss: {np.mean(recon_losses):.4f} | '
                        f'KL Loss: {np.mean(kl_losses)}')

            torch.save(model.module.state_dict(), os.path.join(train_config['task_name'],
                                                              train_config['vae_autoencoder_ckpt_name']))

    print('Done Training...')
    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VAE training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument('--local_rank', type=int, default=0)  # This is for multi-GPU training
    args = parser.parse_args()
    
    train(args.local_rank, args)
