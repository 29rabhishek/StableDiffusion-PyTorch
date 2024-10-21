import numpy as np
import torchvision
from torch.utils.data import Dataset

class ImageNetDatasetEEG(Dataset):
    r"""
    A dataset class for ImageNet images, optimized for batch loading from a .npz file.
    """

    def __init__(self, im_path, im_size, im_channels, 
                 condition_config=None):
        r"""
        Initializes the dataset properties.
        
        :param im_path: Path to the .npz file containing images and labels.
        :param im_size: Size to which images will be resized.
        :param im_channels: Number of channels in the images.
        :param condition_config: Configuration for conditioning types.
        """
        self.im_size = im_size
        self.im_channels = im_channels
        
        # Load data from .npz file using memory mapping
        npz_data = np.load(im_path, mmap_mode='r')
        self.images = npz_data["images"]
        self.labels = npz_data["labels"]
        self.embeddings = npz_data["embeddings"]

        # Conditioning types
        self.condition_types = [] if condition_config is None else condition_config['condition_types']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Set conditioning info
        cond_inputs = {}
        if 'eeg' in self.condition_types:
            cond_inputs['eeg'] = self.embeddings[index]

        # Load image and convert to tensor
        im = self.images[index]
        im_tensor = torchvision.transforms.ToTensor()(im)

        # Normalize image tensor to [-1, 1]
        im_tensor = (2 * im_tensor) - 1

        if len(self.condition_types) == 0:
            return im_tensor
        else:
            return im_tensor, cond_inputs

# Example usage
# dataset = ImageNetDataset('path/to/data.npz', 224, 3)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)