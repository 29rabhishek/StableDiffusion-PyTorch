import yaml
import argparse
import torch
import random
import torchvision
import os
import numpy as np
from models.vae import VAE
from torch.utils.data import DataLoader, DistributedSampler
from dataset.mnist_dataset import MnistDataset
from dataset.celeb_dataset import CelebDataset
from torch.optim import Adam
from torchvision.utils import make_grid
from utils.logger import setup_logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4"
# from pudb import set_trace; set_trace()


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

    
def kl_divergence_loss(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl_loss

def train(rank, world_size, args):
    # Initialize process group for distributed training
    ddp_setup(rank, world_size)


    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)



    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']

    if rank == 0: 
        logger = setup_logger("VAE_logger", save_dir=train_config['task_name'] , if_train=True)

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

        for _, im in enumerate(data_loader):
            step_count += 1
            im = im.float().to(rank)

            with autocast():  # Mixed precision training
                
                model_output = model(im)
                output, mu, logvar = model_output

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

                kl_loss = kl_divergence_loss(mu, logvar)
                kl_losses.append(kl_loss.item())

                g_loss = recon_loss + kl_loss
                losses.append(g_loss.item())

            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()
            optimizer_g.zero_grad()
        print(f'GPU[{rank}] epoch: {epoch_idx+1} | Total Loss: {np.mean(losses)} | Recon Loss : {np.mean(recon_loss):.4f} | KL Loss:  {np.mean(kl_losses)}')
        if rank == 0:  # Log only on the main process      
            torch.save(model.module.state_dict(), f"{os.path.join(train_config['task_name'],train_config['vae_autoencoder_ckpt_name'])}_{epoch_idx+1}.pth")
            logger.info(f"Checkpoint Saved {train_config['vae_autoencoder_ckpt_name']}_{epoch_idx+1}.pth")

    print('Done Training...')
    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for VAE training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    # parser.add_argument('--local_rank', type=int, default=0)  # This is for multi-GPU training
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, args), nprocs = world_size)
