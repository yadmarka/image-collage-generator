# Functions needed:
#- load_source_images(directory)
#- calculate_average_color(image)
#- extract_metadata(image, filepath)
# - categorize_batch(image_list)
# - save_to_cache(data, filepath)
# - load_from_cache(filepath)

"""
Categorize all source images by calculating their average colors.
Creates a data structure optimized for fast color matching.
"""
import json
import os
from typing import List, Dict, Any, Tuple
from PIL import Image
from tqdm import tqdm
import yaml
import numpy as np

from utils.image_loader import get_image_files, load_image, get_image_dimensions
from src.color_analysis import calculate_average_color_optimized, color_to_hex


class SourceImage:
    """Represents a source image with its metadata and average color."""
    
    def __init__(self, filepath: str, avg_color: Tuple[int, int, int], 
                 width: int, height: int):
        """
        Args:
            filepath: Absolute or relative path to the image
            avg_color: Average RGB color tuple (R, G, B)
            width: Image width in pixels
            height: Image height in pixels
        """
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.avg_color = avg_color
        self.width = width
        self.height = height
        self.aspect_ratio = width / height if height != 0 else 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'filepath': self.filepath,
            'filename': self.filename,
            'avg_color_rgb': self.avg_color,
            'avg_color_hex': color_to_hex(self.avg_color),
            'dimensions': {
                'width': self.width,
                'height': self.height
            },
            'aspect_ratio': round(self.aspect_ratio, 3)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceImage':
        """Create SourceImage from dictionary."""
        return cls(
            filepath=data['filepath'],
            avg_color=tuple(data['avg_color_rgb']),
            width=data['dimensions']['width'],
            height=data['dimensions']['height']
        )


class SourceImagePalette:
    """
    Container for all source images, optimized for color matching.
    Provides efficient lookup and filtering methods.
    """
    
    def __init__(self):
        self.images: List[SourceImage] = []
        self._color_array = None  # Cached numpy array of colors for fast matching
    
    def add_image(self, image: SourceImage) -> None:
        """Add a source image to the palette."""
        self.images.append(image)
        self._color_array = None  # Invalidate cache
    
    def get_color_array(self) -> np.ndarray:
        """
        Get numpy array of all colors for vectorized operations.
        Returns array of shape (N, 3) where N is number of images.
        """
        if self._color_array is None:
            self._color_array = np.array([img.avg_color for img in self.images])
        return self._color_array
    
    def find_closest_match(self, target_color: Tuple[int, int, int]) -> SourceImage:
        """
        Find the source image with the closest average color to target.
        Uses Euclidean distance in RGB space.
        
        Args:
            target_color: Target RGB color tuple
        
        Returns:
            SourceImage with the closest matching color
        """
        if not self.images:
            raise ValueError("Palette is empty")
        
        # Convert target to numpy array
        target = np.array(target_color)
        
        # Get all colors as array
        colors = self.get_color_array()
        
        # Calculate Euclidean distances (vectorized)
        distances = np.linalg.norm(colors - target, axis=1)
        
        # Find index of minimum distance
        closest_idx = np.argmin(distances)
        
        return self.images[closest_idx]
    
    def find_closest_matches(self, target_color: Tuple[int, int, int], n: int = 5) -> List[Tuple[SourceImage, float]]:
        """
        Find the N closest matching source images.
        
        Args:
            target_color: Target RGB color tuple
            n: Number of matches to return
        
        Returns:
            List of (SourceImage, distance) tuples, sorted by distance
        """
        if not self.images:
            raise ValueError("Palette is empty")
        
        # Convert target to numpy array
        target = np.array(target_color)
        
        # Get all colors as array
        colors = self.get_color_array()
        
        # Calculate Euclidean distances
        distances = np.linalg.norm(colors - target, axis=1)
        
        # Get indices of n smallest distances
        closest_indices = np.argsort(distances)[:n]
        
        # Return list of (image, distance) tuples
        return [(self.images[idx], distances[idx]) for idx in closest_indices]
    
    def filter_by_aspect_ratio(self, target_ratio: float, tolerance: float = 0.1) -> 'SourceImagePalette':
        """
        Create a new palette with only images matching the target aspect ratio.
        
        Args:
            target_ratio: Desired aspect ratio (width/height)
            tolerance: Allowed difference (e.g., 0.1 means ±10%)
        
        Returns:
            New SourceImagePalette with filtered images
        """
        filtered_palette = SourceImagePalette()
        for img in self.images:
            if abs(img.aspect_ratio - target_ratio) / target_ratio <= tolerance:
                filtered_palette.add_image(img)
        return filtered_palette
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert palette to dictionary for JSON serialization."""
        return {
            'total_images': len(self.images),
            'images': [img.to_dict() for img in self.images]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceImagePalette':
        """Load palette from dictionary."""
        palette = cls()
        for img_data in data['images']:
            palette.add_image(SourceImage.from_dict(img_data))
        return palette
    
    def __len__(self) -> int:
        """Return number of images in palette."""
        return len(self.images)
    
    def __getitem__(self, index: int) -> SourceImage:
        """Allow indexing into the palette."""
        return self.images[index]


def categorize_single_image(filepath: str) -> SourceImage:
    """
    Process a single image and extract its metadata.
    
    Args:
        filepath: Path to the image file
    
    Returns:
        SourceImage object with metadata
    
    Raises:
        Exception if image cannot be processed
    """
    # Load image
    image = load_image(filepath)
    
    # Get dimensions
    width, height = get_image_dimensions(image)
    
    # Calculate average color (optimized for speed)
    avg_color = calculate_average_color_optimized(image)
    
    # Create and return SourceImage object
    return SourceImage(filepath, avg_color, width, height)


def categorize_all_images(image_directory: str, supported_formats: List[str]) -> SourceImagePalette:
    """
    Categorize all images in a directory and build a palette.
    
    Args:
        image_directory: Path to directory containing source images
        supported_formats: List of supported file extensions
    
    Returns:
        SourceImagePalette containing all categorized images
    """
    print(f"Scanning directory: {image_directory}")
    
    # Get all image files
    image_files = get_image_files(image_directory, supported_formats)
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("WARNING: No images found!")
        return SourceImagePalette()
    
    # Create palette
    palette = SourceImagePalette()
    
    # Process each image with progress bar
    print("Categorizing images...")
    failed_count = 0
    
    for filepath in tqdm(image_files, desc="Processing images"):
        try:
            source_image = categorize_single_image(filepath)
            palette.add_image(source_image)
        except Exception as e:
            failed_count += 1
            tqdm.write(f"Failed to process {filepath}: {str(e)}")
    
    print(f"\n✓ Successfully categorized {len(palette)} images")
    if failed_count > 0:
        print(f"✗ Failed to process {failed_count} images")
    
    return palette


def save_palette(palette: SourceImagePalette, output_filepath: str) -> None:
    """
    Save palette to JSON file.
    
    Args:
        palette: SourceImagePalette to save
        output_filepath: Path to save JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Convert to dictionary and save
    data = palette.to_dict()
    
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Palette saved to: {output_filepath}")


def load_palette(filepath: str) -> SourceImagePalette:
    """
    Load palette from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        SourceImagePalette loaded from file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Palette file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    palette = SourceImagePalette.from_dict(data)
    print(f"✓ Loaded palette with {len(palette)} images")
    
    return palette


def print_palette_statistics(palette: SourceImagePalette) -> None:
    """
    Print useful statistics about the palette.
    
    Args:
        palette: SourceImagePalette to analyze
    """
    if len(palette) == 0:
        print("Palette is empty")
        return
    
    print("\n" + "="*60)
    print("PALETTE STATISTICS")
    print("="*60)
    
    print(f"Total images: {len(palette)}")
    
    # Color distribution
    colors = palette.get_color_array()
    print(f"\nColor range:")
    print(f"  R: {colors[:, 0].min()}-{colors[:, 0].max()}")
    print(f"  G: {colors[:, 1].min()}-{colors[:, 1].max()}")
    print(f"  B: {colors[:, 2].min()}-{colors[:, 2].max()}")
    
    # Aspect ratios
    aspect_ratios = [img.aspect_ratio for img in palette.images]
    print(f"\nAspect ratios:")
    print(f"  Min: {min(aspect_ratios):.3f}")
    print(f"  Max: {max(aspect_ratios):.3f}")
    print(f"  Average: {sum(aspect_ratios) / len(aspect_ratios):.3f}")
    
    # Dimensions
    widths = [img.width for img in palette.images]
    heights = [img.height for img in palette.images]
    print(f"\nDimensions:")
    print(f"  Width range: {min(widths)}-{max(widths)} pixels")
    print(f"  Height range: {min(heights)}-{max(heights)} pixels")
    
    # Sample images
    print(f"\nSample images (first 3):")
    for i, img in enumerate(palette.images[:3], 1):
        print(f"{i}. {img.filename}")
        print(f"   Size: {img.width}x{img.height}")
        print(f"   Avg color: RGB{img.avg_color} ({color_to_hex(img.avg_color)})")


def main():
    """
    Main function to categorize all source images.
    """
    print("="*60)
    print("SOURCE IMAGE CATEGORIZATION")
    print("="*60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    source_config = config['source_images']
    
    # Check if cache exists
    cache_file = source_config['cache_file']
    
    if os.path.exists(cache_file):
        print(f"\nCache file found: {cache_file}")
        response = input("Load from cache? (y/n): ").lower()
        
        if response == 'y':
            palette = load_palette(cache_file)
            print_palette_statistics(palette)
            return
    
    # Categorize images
    print()
    palette = categorize_all_images(
        image_directory=source_config['directory'],
        supported_formats=source_config['supported_formats']
    )
    
    if len(palette) == 0:
        print("\nNo images to save. Please add images to", source_config['directory'])
        return
    
    # Save to cache
    save_palette(palette, cache_file)
    
    # Print statistics
    print_palette_statistics(palette)
    
    # Demo: Test color matching
    print("\n" + "="*60)
    print("COLOR MATCHING DEMO")
    print("="*60)
    
    # Test with a few colors
    test_colors = [
        ((255, 0, 0), "Pure Red"),
        ((0, 255, 0), "Pure Green"),
        ((0, 0, 255), "Pure Blue"),
        ((128, 128, 128), "Gray")
    ]
    
    print("\nFinding closest matches for test colors:")
    for color, name in test_colors:
        match = palette.find_closest_match(color)
        print(f"\n{name} RGB{color}:")
        print(f"  → Best match: {match.filename}")
        print(f"     Color: RGB{match.avg_color} ({color_to_hex(match.avg_color)})")


if __name__ == "__main__":
    main()