# Functions needed:
# - load_target_image(filepath)
# - calculate_grid_dimensions(image, section_size)
# - segment_image(image, grid_dims)
# - calculate_section_colors(segments)
# - visualize_segments(segments)

"""
Segment target image into a grid of rectangles and calculate average colors.
User specifies the number of segments (grid dimensions), not the rectangle size.
"""
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
import json
import yaml


class ImageSegment:
    """Represents a single segment of the target image."""
    
    def __init__(self, x: int, y: int, width: int, height: int, avg_color: Tuple[int, int, int]):
        """
        Args:
            x: Top-left x coordinate in the original image
            y: Top-left y coordinate in the original image
            width: Width of the segment
            height: Height of the segment
            avg_color: Average RGB color of this segment
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.avg_color = avg_color
        self.grid_x = None  # Will be set when placed in grid
        self.grid_y = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for JSON serialization."""
        return {
            'position': {'x': self.x, 'y': self.y},
            'dimensions': {'width': self.width, 'height': self.height},
            'grid_position': {'grid_x': self.grid_x, 'grid_y': self.grid_y},
            'avg_color_rgb': self.avg_color,
            'avg_color_hex': '#{:02x}{:02x}{:02x}'.format(*self.avg_color)
        }


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
        - base_width: Width of regular segments
        - base_height: Height of regular segments
        - remainder_width: Width of the last column (0 if divides evenly)
        - remainder_height: Height of the last row (0 if divides evenly)
    """
    # Calculate base segment dimensions (integer division)
    base_width = image_width // num_segments_x
    base_height = image_height // num_segments_y
    
    # Calculate remainder pixels
    remainder_width = image_width % num_segments_x
    remainder_height = image_height % num_segments_y
    
    # Print information
    print(f"Image dimensions: {image_width}x{image_height}")
    print(f"Grid dimensions: {num_segments_x} columns x {num_segments_y} rows")
    print(f"Base segment size: {base_width}x{base_height} pixels")
    
    if remainder_width > 0 or remainder_height > 0:
        print(f"Remainder pixels: {remainder_width}px width, {remainder_height}px height")
        if remainder_width > 0:
            print(f"  → Last column will be {base_width + remainder_width}x{base_height} pixels")
        if remainder_height > 0:
            print(f"  → Last row will be {base_width}x{base_height + remainder_height} pixels")
        if remainder_width > 0 and remainder_height > 0:
            print(f"  → Bottom-right corner will be {base_width + remainder_width}x{base_height + remainder_height} pixels")
    else:
        print("Image divides evenly - no partial rectangles needed!")
    
    return base_width, base_height, remainder_width, remainder_height


def extract_segment_color(image: Image.Image, x: int, y: int, 
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


def segment_image(image: Image.Image, num_segments_x: int, num_segments_y: int) -> Tuple[List[List[ImageSegment]], Dict[str, Any]]:
    """
    Segment an image into a grid with specified number of segments.
    Handles partial rectangles at edges if image doesn't divide evenly.
    
    Args:
        image: PIL Image object (must be in RGB mode)
        num_segments_x: Number of segments horizontally (columns)
        num_segments_y: Number of segments vertically (rows)
    
    Returns:
        Tuple of (2D list of ImageSegment objects, metadata dict)
        - segments[row][col] contains the ImageSegment at that grid position
        - metadata contains information about the segmentation
    """
    width, height = image.size
    
    # Validate inputs
    if num_segments_x <= 0 or num_segments_y <= 0:
        raise ValueError("Number of segments must be positive")
    if num_segments_x > width or num_segments_y > height:
        raise ValueError(f"Cannot create {num_segments_x}x{num_segments_y} segments from {width}x{height} image")
    
    # Calculate segment dimensions
    base_width, base_height, remainder_width, remainder_height = calculate_segment_dimensions(
        width, height, num_segments_x, num_segments_y
    )
    
    print(f"\nCreating {num_segments_x * num_segments_y} segments...")
    
    # Create 2D array to store segments
    segments = []
    
    # Track current y position
    current_y = 0
    
    # Process each row
    for row in range(num_segments_y):
        segment_row = []
        
        # Determine height for this row
        if row == num_segments_y - 1 and remainder_height > 0:
            # Last row gets the remainder pixels
            row_height = base_height + remainder_height
        else:
            row_height = base_height
        
        # Track current x position
        current_x = 0
        
        # Process each column in this row
        for col in range(num_segments_x):
            # Determine width for this column
            if col == num_segments_x - 1 and remainder_width > 0:
                # Last column gets the remainder pixels
                col_width = base_width + remainder_width
            else:
                col_width = base_width
            
            # Extract average color from this region
            avg_color = extract_segment_color(image, current_x, current_y, col_width, row_height)
            
            # Create segment object
            segment = ImageSegment(current_x, current_y, col_width, row_height, avg_color)
            segment.grid_x = col
            segment.grid_y = row
            
            segment_row.append(segment)
            
            # Move to next column position
            current_x += col_width
        
        segments.append(segment_row)
        
        # Move to next row position
        current_y += row_height
    
    # Create metadata with detailed segment size information
    segment_sizes = {}
    for row in range(num_segments_y):
        for col in range(num_segments_x):
            seg = segments[row][col]
            size_key = f"{seg.width}x{seg.height}"
            if size_key not in segment_sizes:
                segment_sizes[size_key] = 0
            segment_sizes[size_key] += 1
    
    metadata = {
        'image_dimensions': {'width': width, 'height': height},
        'grid_size': {'columns': num_segments_x, 'rows': num_segments_y},
        'base_segment_size': {'width': base_width, 'height': base_height},
        'remainder_pixels': {'width': remainder_width, 'height': remainder_height},
        'total_segments': num_segments_x * num_segments_y,
        'segment_size_distribution': segment_sizes
    }
    
    print(f"✓ Created {metadata['total_segments']} segments")
    print(f"  Segment sizes in grid:")
    for size, count in segment_sizes.items():
        print(f"    {size}: {count} segments")
    
    return segments, metadata


def save_segments_to_json(segments: List[List[ImageSegment]], metadata: Dict[str, Any], 
                          output_filepath: str) -> None:
    """
    Save segmented image data to JSON file.
    
    Args:
        segments: 2D list of ImageSegment objects
        metadata: Metadata about the segmentation
        output_filepath: Path to save JSON file
    """
    # Convert segments to serializable format
    segments_data = []
    for row in segments:
        row_data = [seg.to_dict() for seg in row]
        segments_data.append(row_data)
    
    # Combine into output structure
    output_data = {
        'metadata': metadata,
        'segments': segments_data
    }
    
    # Save to JSON
    import os
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Segment data saved to: {output_filepath}")


def visualize_segments(segments: List[List[ImageSegment]], output_path: str) -> None:
    """
    Create a visualization showing the average color of each segment.
    Also draws grid lines to show segment boundaries.
    
    Args:
        segments: 2D list of ImageSegment objects
        output_path: Path to save the visualization image
    """
    import os
    from PIL import ImageDraw
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    num_rows = len(segments)
    num_cols = len(segments[0])
    
    # Calculate total image size from segments
    total_width = sum(seg.width for seg in segments[0])
    total_height = sum(segments[row][0].height for row in range(num_rows))
    
    # Create new image
    visualization = Image.new('RGB', (total_width, total_height))
    draw = ImageDraw.Draw(visualization)
    
    # Fill each segment with its average color
    for row in segments:
        for segment in row:
            # Create a rectangle filled with the average color
            rect = Image.new('RGB', (segment.width, segment.height), segment.avg_color)
            visualization.paste(rect, (segment.x, segment.y))
            
            # Draw grid lines (optional - makes segments more visible)
            draw.rectangle(
                [segment.x, segment.y, segment.x + segment.width - 1, segment.y + segment.height - 1],
                outline=(128, 128, 128),
                width=1
            )
    
    visualization.save(output_path)
    print(f"✓ Visualization saved to: {output_path}")


def main():
    """
    Main function to demonstrate image segmentation with user-specified grid.
    """
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get user input for grid dimensions
    print("="*60)
    print("IMAGE SEGMENTATION - USER-SPECIFIED GRID")
    print("="*60)
    
    # You can either hardcode these or get from user input
    # For now, let's use input (you can change this later)
    try:
        num_segments_x = int(input("\nEnter number of segments horizontally (width): "))
        num_segments_y = int(input("Enter number of segments vertically (height): "))
    except ValueError:
        print("Invalid input! Using default: 40x30")
        num_segments_x = 40
        num_segments_y = 30
    
    # Example: Load a target image
    target_image_path = "data/target_images/example.jpg"  # Change this to your image
    
    try:
        from utils.image_loader import load_image
        
        print(f"\nLoading target image: {target_image_path}")
        image = load_image(target_image_path)
        print(f"✓ Image loaded: {image.size[0]}x{image.size[1]} pixels\n")
        
        # Segment the image
        print("="*60)
        print("SEGMENTING IMAGE")
        print("="*60)
        segments, metadata = segment_image(image, num_segments_x, num_segments_y)
        
        # Save to JSON
        output_json_path = "cache/target_segments.json"
        save_segments_to_json(segments, metadata, output_json_path)
        
        # Create visualization
        viz_path = "output/collages/segments_visualization.png"
        print("\nCreating visualization...")
        visualize_segments(segments, viz_path)
        
        # Print sample data
        print("\n" + "="*60)
        print("SAMPLE SEGMENTS (corners of the grid)")
        print("="*60)
        
        # Show corners to demonstrate different sizes
        corners = [
            (0, 0, "Top-Left"),
            (0, len(segments[0])-1, "Top-Right"),
            (len(segments)-1, 0, "Bottom-Left"),
            (len(segments)-1, len(segments[0])-1, "Bottom-Right")
        ]
        
        for row_idx, col_idx, label in corners:
            seg = segments[row_idx][col_idx]
            print(f"\n{label} [{row_idx},{col_idx}]:")
            print(f"  Position: ({seg.x}, {seg.y})")
            print(f"  Size: {seg.width}x{seg.height} pixels")
            print(f"  Color: RGB{seg.avg_color} | Hex: {seg.to_dict()['avg_color_hex']}")
        
        print("\n" + "="*60)
        print("✓ SEGMENTATION COMPLETE!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\nError: Target image not found at {target_image_path}")
        print("Please place an image in data/target_images/ and update the path in main()")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()