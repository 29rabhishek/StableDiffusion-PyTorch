from models.vae import VAE
import torch 
import yaml

if __name__ == '__main__':
    config_file = 'config/imagenet.yaml'
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # # parser = argparse.ArgumentParser(description='Arguments for VAE training')
    # parser.add_argument('--config', dest='config_path', default='config/imagenet.yaml', type=str)
    # args = parser.parse_args()
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    model = VAE(im_channels=dataset_config['im_channels'], model_config=autoencoder_config)
    dummy_input = torch.randn(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image
    output, mu, logvar = model(dummy_input)
    print(output.shape, mu.shape, logvar.shape)





