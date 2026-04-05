from PIL import Image
import numpy as np
from src.categorize_images import SourceImage, SourceImagePalette, categorize_all_images


def render_collage(target_image: Image.Image,
                   palette: SourceImagePalette,
                   tile_size: int = 40,
                   method: str = "euclidean") -> Image.Image:

    # Crop to clean tile grid
    width, height = target_image.size
    width = (width // tile_size) * tile_size
    height = (height // tile_size) * tile_size
    target_image = target_image.crop((0, 0, width, height))
    
    #create new empty image
    mosaic = Image.new("RGB", (width, height))

    #loop through each tile on grid
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Extract tile and compute average color
            tile = target_image.crop((x, y, x + tile_size, y + tile_size))
            avg_color = tuple(np.array(tile).mean(axis=(0,1)).astype(int))
            # Use your existing matching function
            match = palette.find_closest_match(avg_color)
            # Load image from filepath
            source_img = Image.open(match.filepath).convert("RGB")
            # Resize to tile
            source_img = source_img.resize((tile_size, tile_size),
                Image.Resampling.LANCZOS)

            #paste into collage
            mosaic.paste(source_img, (x, y))
    return mosaic

# Render collage for all images in the dataset
palette = categorize_all_images(
   image_directory="data/source_images",
   supported_formats=[".jpg", ".jpeg", ".png"]
)
print("Loaded images:", len(palette))
target_image = Image.open("data/target_images/IMG_0092.jpeg").convert("RGB")
collage = render_collage(target_image, palette, tile_size=40)
collage.save("collage.jpg")