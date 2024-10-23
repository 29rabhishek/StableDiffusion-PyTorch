import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_eeg_cond_base import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from utils.config_utils import *
from utils.eeg_utils import *
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, diffusion_model_config,
           autoencoder_model_config, diffusion_config, dataset_config, vae, label):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    im_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    
    ########### Sample random noise latent ##########
    # For not fixing generation with one sample
    xt = torch.randn((1,
                      autoencoder_model_config['z_channels'],
                      im_size,
                      im_size)).to(device)
    ###############################################
    
    ############ Create Conditional Text input ###############
    # text_prompt = ['She is a woman with blond hair. She is wearing lipstick.']
    # neg_prompt = ['He is a man.']
    # empty_prompt = ['']
    # text_prompt_embed = get_text_representation(text_prompt,
    #                                             text_tokenizer,
    #                                             text_model, 
    #                                             device) # giving text embeddings
    # # Can replace empty prompt with negative prompt
    # empty_text_embed = get_text_representation(empty_prompt, text_tokenizer, text_model, device)
    # assert empty_text_embed.shape == text_prompt_embed.shape
    
    # uncond_input = {
    #     'text': empty_text_embed
    # }
    # cond_input = {
    #     'text': text_prompt_embed
    # }
    ###############################################




    ############ Create Conditional Text input ###############
    cond_input, orginal_image = get_eeg_cond_input(label)
    ##########################################################
    
    # By default classifier free guidance is disabled
    # Change value in config or change default value here to enable it
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    save_image_from_tensor(orginal_image)
    ################# Sampling Loop ########################
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        t = (torch.ones((xt.shape[0],)) * i).long().to(device)
        # noise_pred_cond = model(xt, t, cond_input)
        
        # if cf_guidance_scale > 1:
        #     noise_pred_uncond = model(xt, t, uncond_input)
        #     noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred_cond - noise_pred_uncond)
        # else:
        #     noise_pred = noise_pred_cond
        
        # Use scheduler to get x0 and xt-1
        noise_pred = model(xt, t, cond_input)
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final iamge to save time
            ims = vae.decode(xt)
        else:
            ims = x0_pred
        
        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=1)
        img = torchvision.transforms.ToPILImage()(grid)
        
        if not os.path.exists(os.path.join(train_config['task_name'], 'cond_eeg_samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'cond_eeg_samples'))
        img.save(os.path.join(train_config['task_name'], 'cond_eeg_samples', 'x0_label_{}_{}.png'.format(label,i)))
        img.close()
    ##############################################################
    save_image_from_tensor(orginal_image, file_path=os.path.join(train_config['task_name'], 'cond_eeg_samples', f'Orginal_Image_label_{label}.png'))

def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    #added class label to guide it
    config['label'] = args.classlabel
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    # text_tokenizer = None
    # text_model = None
    
    ############# Validate the config #################
    # condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    # assert condition_config is not None, ("This sampling script is for text conditional "
    #                                       "but no conditioning config found")
    # condition_types = get_config_value(condition_config, 'condition_types', [])
    # assert 'text' in condition_types, ("This sampling script is for text conditional "
    #                                     "but no text condition found in config")
    # validate_text_config(condition_config)
    ###############################################
    
    ############# Load tokenizer and text model #################
    # with torch.no_grad():
    #     # Load tokenizer and text model based on config
    #     # Also get empty text representation
    #     text_tokenizer, text_model = get_tokenizer_and_model(condition_config['text_condition_config']
    #                                                          ['text_embed_model'], device=device)
    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_ckpt_name']),
                                         map_location=device))
    else:
        raise Exception('Model checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                              train_config['ldm_ckpt_name'])))
    #####################################
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device), strict=True)
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                          train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, config['label']) ### EEG Embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation with only '
                                                 'text conditioning')
    parser.add_argument('--config', dest='config_path',
                        default='config/imagenet_eeg_cond.yaml', type=str)
    parser.add_argument('--classlabel', default=2, type=int)
    args = parser.parse_args()
    infer(args)
