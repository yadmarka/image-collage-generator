"""
Color analysis utilities for calculating average colors
"""
from PIL import Image
import numpy as np
from typing import Tuple


def calculate_average_color(image: Image.Image) -> Tuple[int, int, int]:
    """
    Calculate the average RGB color of an image.
    
    Args:
        image: PIL Image object in RGB mode
    
    Returns:
        Tuple of (R, G, B) values as integers (0-255)
    """
    # Convert image to numpy array
    img_array = np.array(image)
    
    # Calculate mean across all pixels for each channel
    # Shape is (height, width, 3) -> we want mean of axis 0 and 1
    avg_color = img_array.mean(axis=(0, 1))
    
    # Round and convert to integers
    return tuple(int(c) for c in avg_color)


def calculate_average_color_optimized(image: Image.Image, max_dimension: int = 100) -> Tuple[int, int, int]:
    """
    Calculate average color of an image with optimization for large images.
    Resizes image to max_dimension before calculation for speed.
    
    Args:
        image: PIL Image object in RGB mode
        max_dimension: Maximum width/height for resized image
    
    Returns:
        Tuple of (R, G, B) values as integers (0-255)
    """
    # Resize image if it's too large (for performance)
    width, height = image.size
    if max(width, height) > max_dimension:
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return calculate_average_color(image)


def color_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string.
    
    Args:
        rgb: Tuple of (R, G, B) values
    
    Returns:
        Hex color string (e.g., '#FF5733')
    """
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def hex_to_color(hex_string: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.
    
    Args:
        hex_string: Hex color string (e.g., '#FF5733' or 'FF5733')
    
    Returns:
        Tuple of (R, G, B) values
    """
    hex_string = hex_string.lstrip('#')
    return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))