"""
Image loading and validation utilities
"""
from PIL import Image
import os
from typing import List, Tuple

def get_image_files(directory: str, supported_formats: List[str]) -> List[str]:
    """
    Get all image files from a directory.
    
    Args:
        directory: Path to directory containing images
        supported_formats: List of supported file extensions (e.g., ['.jpg', '.png'])
    
    Returns:
        List of full file paths to images
    """
    #TO DO: Implement a function to get image files from a directory
    '''image_files = []
    with open(os.path.dirname(os.path.realpath( directory ))) as f:
        print('hello')
        print(f)
        if f in supported_formats:
            image_files.append(f)
    '''
    if not os.path.exists(directory):
       raise FileNotFoundError(f"Directory not found: {directory}")
    image_files = []
    for root, dirs, files in os.walk(directory):
       for file in files:
           if any(file.lower().endswith(fmt) for fmt in supported_formats):
               image_files.append(os.path.join(root, file))
    
    return sorted(image_files)

def load_image(filepath: str) -> Image.Image:
    """
    Load an image from filepath and convert to RGB.
    
    Args:
        filepath: Path to image file
    
    Returns:
        PIL Image object in RGB mode
    """
    #Implement image loading and conversion to RBG
    try:
        img = Image.open(filepath)
        # Convert to RGB (handles RGBA, grayscale, etc.)
        return img.convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image {filepath}: {str(e)}")

def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """
    Get image dimensions.
    
    Args:
        image: PIL Image object
    
    Returns:
        Tuple of (width, height)
    """
    #TO DO: Implement a function to get image dimensions (Should be one line)
    return (image.size)
def validate_image(filepath: str) -> bool:
    """
    Check if a file is a valid image.
    
    Args:
        filepath: Path to image file
    
    Returns:
        True if valid image, False otherwise
    """
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False
