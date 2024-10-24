import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from models.vae import VAE
from torch.utils.data import DataLoader
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from dataset.imagenet_dataset import ImageNetDataset

from torch.optim import Adam
from torchvision.utils import make_grid
from utils.logger import setup_logger
from torch.cuda.amp import GradScaler, autocast

def kl_divergence_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    logger = setup_logger("VAE_logger", file_name='vae_logger_24th_oct', save_dir=train_config['task_name'], if_train=True)

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config).to(device)

    im_dataset_cls = {
        'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'imagenet': ImageNetDataset
    }.get(dataset_config['name'])

    im_dataset = im_dataset_cls(split='train',
                                im_path=dataset_config['im_path'],
                                im_size=dataset_config['im_size'],
                                im_channels=dataset_config['im_channels'])

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['autoencoder_batch_size'],
                             shuffle=True)

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    #making path to save vae_generated samples
    if not os.path.exists(os.path.join(train_config['task_name'], 'vae_autoencoder_samples')):
        os.mkdir(os.path.join(train_config['task_name'], 'vae_autoencoder_samples'))

    num_epochs = train_config['autoencoder_epochs']
    recon_criterion = torch.nn.MSELoss(reduce="sum")

    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))

    scaler = torch.cuda.amp.GradScaler()
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0

    model.train()
    for epoch_idx in range(num_epochs):
        recon_losses = 0
        kl_losses = 0
        losses = 0

        for _, im in enumerate(data_loader):
            im = im.float().to(device)

            # with torch.cuda.amp.autocast():  # Mixed precision training
            model_output = model(im)
            output, mu, logvar = model_output

            recon_loss = recon_criterion(output, im)
            recon_losses += recon_loss

            kl_loss = kl_divergence_loss(mu, logvar)
            kl_losses += kl_loss

            g_loss = recon_loss + kl_loss
            losses += g_loss

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

            # scaler.scale(g_loss).backward()
            # scaler.step(optimizer_g)
            # scaler.update()
            # optimizer_g.zero_grad()

        recon_losses = recon_losses / len(data_loader)
        kl_losses = kl_losses / len(data_loader)
        losses = losses / len(data_loader)
        
        if img_save_count % image_save_steps == 0 or img_save_count == 0:
            sample_size = min(8, im.shape[0])
            save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
            save_output = ((save_output + 1) / 2)
            save_input = ((im[:sample_size] + 1) / 2).detach().cpu()

            grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
            img = torchvision.transforms.ToPILImage()(grid)
            img_save_path = os.path.join(train_config['task_name'], 'vae_autoencoder_samples')
            # os.makedirs(img_save_path, exist_ok=True)
            img.save(os.path.join(img_save_path, f'current_autoencoder_sample_{img_save_count}.png'))
            img_save_count += 1
            img.close()

        print(f'Epoch: {epoch_idx + 1} | Total Loss: {losses} | Recon Loss: {recon_loss:.4f} | KL Loss: {kl_losses}')

        torch.save(model.state_dict(), f"{os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])}_{epoch_idx + 1}.pth")
        logger.info(f"Checkpoint Saved {train_config['vae_autoencoder_ckpt_name']}_{epoch_idx + 1}.pth")

    print('Done Training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VAE training')
    parser.add_argument('--config', dest='config_path', default='config/imagenet.yaml', type=str)
    args = parser.parse_args()

    train(args)
