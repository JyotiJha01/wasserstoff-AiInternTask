import os
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the input image for segmentation.
    """
    image = Image.open(image_path).convert("RGB")
    
    # Resize image if it's too large
    max_size = 1024
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    return image_array, image

def get_image_paths(input_dir):
    """
    Get all image file paths from the input directory.
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and 
                   os.path.splitext(f)[1].lower() in valid_extensions]
    return image_paths