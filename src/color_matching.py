# Functions needed:
import numpy as np
from typing import Tuple, List
from categorize_images import SourceImagePalette

def euclidean_distance(color1: Tuple[int, int, int],
                       color2: Tuple[int, int, int]) -> float:
    """
    Compute Euclidean distance between two RGB colors.
    """
    c1 = np.array(color1, dtype=np.float32)
    c2 = np.array(color2, dtype=np.float32)
    return float(np.linalg.norm(c1 - c2))

def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB (0-255) to XYZ.
    """
    rgb = rgb / 255.0

    # sRGB companding
    mask = rgb > 0.04045
    rgb = np.where(mask,
                   ((rgb + 0.055) / 1.055) ** 2.4,
                   rgb / 12.92)

    rgb *= 100

    # Observer = 2°, Illuminant = D65
    x = rgb[0] * 0.4124 + rgb[1] * 0.3576 + rgb[2] * 0.1805
    y = rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
    z = rgb[0] * 0.0193 + rgb[1] * 0.1192 + rgb[2] * 0.9505

    return np.array([x, y, z])


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """
    Convert XYZ to LAB.
    """
    # Reference white (D65)
    ref = np.array([95.047, 100.000, 108.883])
    xyz = xyz / ref

    mask = xyz > 0.008856
    xyz = np.where(mask,
                   xyz ** (1/3),
                   (7.787 * xyz) + (16 / 116))

    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])

    return np.array([L, a, b])


def delta_e_distance(color1: Tuple[int, int, int],
                     color2: Tuple[int, int, int]) -> float:
    """
    Compute CIE76 Delta E distance between two RGB colors.
    More perceptually accurate than RGB Euclidean.
    """
    rgb1 = np.array(color1, dtype=np.float32)
    rgb2 = np.array(color2, dtype=np.float32)

    lab1 = xyz_to_lab(rgb_to_xyz(rgb1))
    lab2 = xyz_to_lab(rgb_to_xyz(rgb2))

    return float(np.linalg.norm(lab1 - lab2))

def find_best_match(target_color: Tuple[int, int, int],
                    source_palette,
                    method: str = "delta_e"):
    """
    Find best matching SourceImage in palette.
    
    method:
        'euclidean'  → RGB distance
        'delta_e'    → LAB perceptual distance
    """
    if not source_palette.images:
        raise ValueError("Palette is empty")

    best_image = None
    best_distance = float("inf")

    for img in source_palette.images:
        if method == "euclidean":
            dist = euclidean_distance(target_color, img.avg_color)
        elif method == "delta_e":
            dist = delta_e_distance(target_color, img.avg_color)
        else:
            raise ValueError("Unknown method")

        if dist < best_distance:
            best_distance = dist
            best_image = img

    return best_image

def match_all_sections(target_sections: List,
                       source_palette,
                       method: str = "delta_e"):
    """
    Match each target section to the best source image.

    Returns:
        List of (section, matched_source_image)
    """

    matches = []

    for section in target_sections:
        # If section is already a color tuple
        if isinstance(section, tuple):
            target_color = section
        else:
            target_color = section.avg_color

        best_match = find_best_match(
            target_color,
            source_palette,
            method=method
        )

        matches.append((section, best_match))

    return matches