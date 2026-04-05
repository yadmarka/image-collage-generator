"""
Extract average colors from target image sections.
Creates a data structure optimized for matching with source images.
"""
import json
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import yaml

from utils.image_loader import load_image

#TO DO
class TargetSection:
    """Represents a single section of the target image with its average color."""
    
    def __init__(self, grid_x: int, grid_y: int, x: int, y: int, 
                 width: int, height: int, avg_color: Tuple[int, int, int]):
        """
        Args:
            grid_x: Column position in grid (0-indexed)
            grid_y: Row position in grid (0-indexed)
            x: Top-left x coordinate in original image
            y: Top-left y coordinate in original image
            width: Width of the section
            height: Height of the section
            avg_color: Average RGB color tuple (R, G, B)
        """
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.avg_color = avg_color
        self.matched_image = None  # Will be filled in Step 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'grid_position': {
                'x': self.grid_x,
                'y': self.grid_y
            },
            'pixel_position': {
                'x': self.x,
                'y': self.y
            },
            'dimensions': {
                'width': self.width,
                'height': self.height
            },
            'avg_color_rgb': self.avg_color,
            'avg_color_hex': '#{:02x}{:02x}{:02x}'.format(*self.avg_color),
            'matched_image': self.matched_image  # For Step 4
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetSection':
        """Create TargetSection from dictionary."""
        section = cls(
            grid_x=data['grid_position']['x'],
            grid_y=data['grid_position']['y'],
            x=data['pixel_position']['x'],
            y=data['pixel_position']['y'],
            width=data['dimensions']['width'],
            height=data['dimensions']['height'],
            avg_color=tuple(data['avg_color_rgb'])
        )
        section.matched_image = data.get('matched_image')
        return section


class TargetGrid:
    """
    Represents the entire target image as a 2D grid of sections.
    Optimized for efficient color matching operations.
    """
    
    def __init__(self, num_cols: int, num_rows: int):
        """
        Args:
            num_cols: Number of columns in the grid
            num_rows: Number of rows in the grid
        """
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.sections: List[List[TargetSection]] = []
        self._color_array = None  # Cached for vectorized operations
    
    def add_row(self, row: List[TargetSection]) -> None:
        """Add a row of sections to the grid."""
        if len(row) != self.num_cols:
            raise ValueError(f"Row must have {self.num_cols} sections")
        self.sections.append(row)
        self._color_array = None  # Invalidate cache
    
    def get_section(self, grid_x: int, grid_y: int) -> TargetSection:
        """
        Get section at specific grid position.
        
        Args:
            grid_x: Column index (0-indexed)
            grid_y: Row index (0-indexed)
        
        Returns:
            TargetSection at that position
        """
        if grid_y < 0 or grid_y >= self.num_rows or grid_x < 0 or grid_x >= self.num_cols:
            raise IndexError(f"Grid position ({grid_x}, {grid_y}) out of bounds")
        return self.sections[grid_y][grid_x]
    
    def get_all_colors(self) -> np.ndarray:
        """
        Get numpy array of all section colors for vectorized operations.
        Returns array of shape (num_rows, num_cols, 3).
        """
        if self._color_array is None:
            colors = []
            for row in self.sections:
                row_colors = [section.avg_color for section in row]
                colors.append(row_colors)
            self._color_array = np.array(colors)
        return self._color_array
    
    def get_flattened_colors(self) -> np.ndarray:
        """
        Get flattened array of all colors.
        Returns array of shape (total_sections, 3).
        Useful for batch operations.
        """
        colors = self.get_all_colors()
        return colors.reshape(-1, 3)
    
    def get_flattened_sections(self) -> List[TargetSection]:
        """
        Get all sections as a flat list.
        Useful for iterating over all sections.
        """
        flat_sections = []
        for row in self.sections:
            flat_sections.extend(row)
        return flat_sections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert grid to dictionary for JSON serialization."""
        return {
            'grid_dimensions': {
                'columns': self.num_cols,
                'rows': self.num_rows
            },
            'total_sections': self.num_cols * self.num_rows,
            'sections': [[section.to_dict() for section in row] for row in self.sections]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TargetGrid':
        """Load grid from dictionary."""
        grid = cls(
            num_cols=data['grid_dimensions']['columns'],
            num_rows=data['grid_dimensions']['rows']
        )
        for row_data in data['sections']:
            row = [TargetSection.from_dict(section_data) for section_data in row_data]
            grid.add_row(row)
        return grid
    
    def __len__(self) -> int:
        """Return total number of sections."""
        return self.num_cols * self.num_rows


def calculate_segment_dimensions(image_width: int, image_height: int,
                                 num_segments_x: int, num_segments_y: int) -> Tuple[int, int, int, int]:
    """
    Calculate the dimensions of regular and remainder segments.
    
    Args:
        image_width: Width of the target image
        image_height: Height of the target image
        num_segments_x: Number of segments horizontally (columns)
        num_segments_y: Number of segments vertically (rows)
    
    Returns:
        Tuple of (base_width, base_height, remainder_width, remainder_height)
    """
    #TO DO: Calculate the dimensions of each segment, accounting for any remainder pixels
    
    base_width = image_width // num_segments_x

    base_height = image_height // num_segments_y

    remainder_width = image_width % num_segments_x

    remainder_height = image_width % num_segments_x
    
    return (base_width, base_height, remainder_width, remainder_height)



def extract_section_color(image: Image.Image, x: int, y: int, 
                          width: int, height: int) -> Tuple[int, int, int]:
    """
    Extract average color from a specific region of the image.
    
    Args:
        image: PIL Image object
        x: Top-left x coordinate
        y: Top-left y coordinate
        width: Width of the region
        height: Height of the region
    
    Returns:
        Tuple of (R, G, B) average color
    """
    # Crop the region
    region = image.crop((x, y, x + width, y + height))
    
    # Convert to numpy array and calculate mean
    region_array = np.array(region)
    avg_color = region_array.mean(axis=(0, 1))
    
    return tuple(int(c) for c in avg_color)



def extract_target_colors(image: Image.Image, num_segments_x: int, 
                          num_segments_y: int) -> TargetGrid:
    """
    Extract average colors from all sections of the target image.
    
    Args:
        image: PIL Image object (must be in RGB mode)
        num_segments_x: Number of segments horizontally (columns)
        num_segments_y: Number of segments vertically (rows)
    
    Returns:
        TargetGrid containing all sections with their colors
    """
    # TO DO: Use the calculated segment dimensions to loop through the image and extract colors for each section, building the TargetGrid
    width, height = image.size
    if num_segments_x <= 0 or num_segments_y <= 0:
        raise ValueError("Number of segments must be positive")
    if num_segments_x > width or num_segments_y > height:
        raise ValueError(f"Cannot create {num_segments_x}x{num_segments_y} segments from {width}x{height} image")
    
    print(f"Extracting colors from {width}x{height} image")
    print(f"Creating {num_segments_x}x{num_segments_y} grid ({num_segments_x * num_segments_y} sections)")
    
    # Calculate segment dimensions
    base_width, base_height, remainder_width, remainder_height = calculate_segment_dimensions(
        width, height, num_segments_x, num_segments_y
    )
    
    print(f"Base section size: {base_width}x{base_height} pixels")
    if remainder_width > 0 or remainder_height > 0:
        print(f"Remainder pixels: {remainder_width}px width, {remainder_height}px height")
    
    # Create grid
    grid = TargetGrid(num_segments_x, num_segments_y)
    
    # Track current y position
    current_y = 0
    
    # Process each row
    print("\nExtracting section colors...")
    for row_idx in range(num_segments_y):
        row_sections = []
        
        # Determine height for this row
        if row_idx == num_segments_y - 1 and remainder_height > 0:
            row_height = base_height + remainder_height
        else:
            row_height = base_height
        
        # Track current x position
        current_x = 0
        
        # Process each column in this row
        for col_idx in range(num_segments_x):
            # Determine width for this column
            if col_idx == num_segments_x - 1 and remainder_width > 0:
                col_width = base_width + remainder_width
            else:
                col_width = base_width
            
            # Extract average color from this region
            avg_color = extract_section_color(image, current_x, current_y, col_width, row_height)
            
            # Create section object
            section = TargetSection(
                grid_x=col_idx,
                grid_y=row_idx,
                x=current_x,
                y=current_y,
                width=col_width,
                height=row_height,
                avg_color=avg_color
            )
            
            row_sections.append(section)
            
            # Move to next column position
            current_x += col_width
        
        # Add row to grid
        grid.add_row(row_sections)
        
        # Move to next row position
        current_y += row_height
        
        # Progress indicator
        if (row_idx + 1) % max(1, num_segments_y // 10) == 0:
            progress = (row_idx + 1) / num_segments_y * 100
            print(f"  Progress: {progress:.0f}% ({row_idx + 1}/{num_segments_y} rows)")
    
    print(f"✓ Extracted colors from {len(grid)} sections")
    
    return grid

def save_target_grid(grid: TargetGrid, output_filepath: str) -> None:
    """
    Save target grid to JSON file.
    
    Args:
        grid: TargetGrid to save
        output_filepath: Path to save JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Convert to dictionary and save
    data = grid.to_dict()
    
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Target grid saved to: {output_filepath}")


def load_target_grid(filepath: str) -> TargetGrid:
    """
    Load target grid from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        TargetGrid loaded from file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target grid file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    grid = TargetGrid.from_dict(data)
    print(f"✓ Loaded target grid with {len(grid)} sections")
    
    return grid


def visualize_target_grid(grid: TargetGrid, output_path: str) -> None:
    """
    Create a visualization showing the average color of each section.
    
    Args:
        grid: TargetGrid to visualize
        output_path: Path to save the visualization image
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate total image size from sections
    total_width = sum(grid.sections[0][col].width for col in range(grid.num_cols))
    total_height = sum(grid.sections[row][0].height for row in range(grid.num_rows))
    
    # Create new image
    visualization = Image.new('RGB', (total_width, total_height))
    
    # Fill each section with its average color
    for row in grid.sections:
        for section in row:
            # Create a rectangle filled with the average color
            rect = Image.new('RGB', (section.width, section.height), section.avg_color)
            visualization.paste(rect, (section.x, section.y))
    
    visualization.save(output_path)
    print(f"✓ Visualization saved to: {output_path}")


def print_grid_statistics(grid: TargetGrid) -> None:
    """
    Print useful statistics about the target grid.
    
    Args:
        grid: TargetGrid to analyze
    """
    print("\n" + "="*60)
    print("TARGET GRID STATISTICS")
    print("="*60)
    
    print(f"Grid dimensions: {grid.num_cols} columns x {grid.num_rows} rows")
    print(f"Total sections: {len(grid)}")
    
    # Color distribution
    colors = grid.get_flattened_colors()
    print(f"\nColor range across all sections:")
    print(f"  R: {colors[:, 0].min()}-{colors[:, 0].max()}")
    print(f"  G: {colors[:, 1].min()}-{colors[:, 1].max()}")
    print(f"  B: {colors[:, 2].min()}-{colors[:, 2].max()}")
    
    # Section sizes
    sections = grid.get_flattened_sections()
    widths = [s.width for s in sections]
    heights = [s.height for s in sections]
    unique_sizes = set((s.width, s.height) for s in sections)
    
    print(f"\nSection sizes:")
    if len(unique_sizes) == 1:
        w, h = list(unique_sizes)[0]
        print(f"  All sections: {w}x{h} pixels")
    else:
        print(f"  Width range: {min(widths)}-{max(widths)} pixels")
        print(f"  Height range: {min(heights)}-{max(heights)} pixels")
        print(f"  Unique sizes: {len(unique_sizes)}")
        for w, h in sorted(unique_sizes):
            count = sum(1 for s in sections if s.width == w and s.height == h)
            print(f"    {w}x{h}: {count} sections")
    
    # Sample sections (corners)
    print(f"\nSample sections (corners):")
    corners = [
        (0, 0, "Top-Left"),
        (grid.num_cols - 1, 0, "Top-Right"),
        (0, grid.num_rows - 1, "Bottom-Left"),
        (grid.num_cols - 1, grid.num_rows - 1, "Bottom-Right")
    ]
    
    for col, row, label in corners:
        section = grid.get_section(col, row)
        hex_color = '#{:02x}{:02x}{:02x}'.format(*section.avg_color)
        print(f"  {label} [{row},{col}]: RGB{section.avg_color} ({hex_color})")


def main():
    """
    Main function to extract colors from target image.
    """
    print("="*60)
    print("TARGET IMAGE COLOR EXTRACTION")
    print("="*60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get grid dimensions
    num_segments_x = config['collage']['grid_segments_x']
    num_segments_y = config['collage']['grid_segments_y']
    
    # Get target image path
    target_image_path = input("\nEnter path to target image (or press Enter for default): ").strip()
    if not target_image_path:
        target_image_path = "data/target_images/example.jpg"
    
    try:
        # Load image
        print(f"\nLoading target image: {target_image_path}")
        image = load_image(target_image_path)
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]} pixels")
        
        # Extract colors
        print("\n" + "="*60)
        print("EXTRACTING SECTION COLORS")
        print("="*60)
        grid = extract_target_colors(image, num_segments_x, num_segments_y)
        
        # Save to JSON
        output_json_path = "cache/target_grid.json"
        save_target_grid(grid, output_json_path)
        
        # Create visualization
        viz_path = "output/collages/target_grid_visualization.png"
        print("\nCreating visualization...")
        visualize_target_grid(grid, viz_path)
        
        # Print statistics
        print_grid_statistics(grid)
        
        print("\n" + "="*60)
        print("✓ COLOR EXTRACTION COMPLETE!")
        print("="*60)
        print(f"\nData saved to: {output_json_path}")
        print(f"Visualization saved to: {viz_path}")
        print(f"\nThis grid is ready for matching in Step 4!")
        
    except FileNotFoundError:
        print(f"\n✗ Error: Target image not found at {target_image_path}")
        print("Please place an image in data/target_images/")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()