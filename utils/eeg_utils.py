import numpy as np
from PIL import Image
import random

def get_eeg_cond_input(label, file_name = "./data/image_eeg_embed/test_embeddings.npz"):
    # valid top 10 labels {1, 38, 9, 10, 11, 12, 16, 18, 25, 26}
    npz_data = np.load(file_name, mmap_mode='r')
    images = npz_data["images"]
    labels = npz_data["labels"]
    embeddings = npz_data["embeddings"]
    # Find indices of embeddings that match the given label
    matching_indices = np.where(labels == label)[0]
    
    if len(matching_indices) == 0:
        raise ValueError(f"No embeddings found for label {label}.")
    
    # Randomly select one index from the matching indices
    random_index = random.choice(matching_indices)

    
    return embeddings[random_index], images[random_index]



def save_image_from_tensor(tensor, file_path):
    """
    Save a NumPy tensor as an image file.
    
    :param tensor: A NumPy array representing the image data. 
                   Should be in the format (H, W, C) for RGB images.
    :param file_path: The path where the image will be saved.
    """
    # Ensure the tensor is in the correct format (H, W, C)
    if tensor.ndim == 2:  # Grayscale image
        image = Image.fromarray(tensor.astype(np.uint8), mode='L')
    elif tensor.ndim == 3 and tensor.shape[2] == 3:  # RGB image
        image = Image.fromarray(tensor.astype(np.uint8), mode='RGB')
    elif tensor.ndim == 3 and tensor.shape[2] == 4:  # RGBA image
        image = Image.fromarray(tensor.astype(np.uint8), mode='RGBA')
    else:
        raise ValueError("Input tensor must be a 2D or 3D array with appropriate channels.")

    # Save the image to the specified file path
    image.save(file_path)

# # Example usage
# # Create a random RGB image as a NumPy array
# image_tensor = np.random.rand(256, 256, 3)  # Random image with values between [0, 1]
# image_tensor = (image_tensor * 255).astype(np.uint8)  # Convert to uint8 format

# # Save the image
# save_image_from_tensor(image_tensor, 'random_image.png')
