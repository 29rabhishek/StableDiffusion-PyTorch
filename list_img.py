import os
import csv
import shutil
from PIL import Image

def list_images_to_csv(base_dir, output_csv, grayscale_dir):
    # Ensure the grayscale directory exists
    if not os.path.exists(grayscale_dir):
        os.makedirs(grayscale_dir)
    
    # Open (or create) the CSV file in write mode
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['File Path', 'Height', 'Width', 'Channels'])
        
        # Walk through the directory
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                
                try:
                    with Image.open(filepath) as img:
                        width, height = img.size
                        channels = len(img.getbands())

                        # Check if the image is grayscale (1 channel)
                        if channels == 1:
                            # Move the grayscale image to the specified directory
                            new_filepath = os.path.join(grayscale_dir, filename)
                            shutil.move(filepath, new_filepath)
                            filepath = new_filepath
                        
                        # Write the file path, height, width, and number of channels to the CSV file
                        writer.writerow([filepath, height, width, channels])
                
                except Exception as e:
                    print(f"Could not process {filepath}: {e}")

if __name__ == "__main__":
    base_directory = "./data/imagenet"  # Replace with the path to your directory
    output_csv_file = "image_shapes.csv"        # Replace with the desired output CSV file name
    grayscale_directory = "./data/gryscale_imagenet"  # Replace with the path to move grayscale images
    list_images_to_csv(base_directory, output_csv_file, grayscale_directory)
