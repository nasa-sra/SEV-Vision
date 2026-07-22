import cv2
import numpy as np
from pathlib import Path

def get_color_to_class_conversion_table(colorsTxtPath):
    # Create a dictionary converting RGB colors to class strings
    color2class = {}
    with open(colorsTxtPath, 'r') as f:
        for line in f:
            line = line.strip()
            # Make sure to ignore the first line ("Category r g b")
            if not line or line.startswith('#') or line.startswith('Category'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            class_name = parts[0]
            r, g, b = map(int, parts[1:4])
            color2class[(r, g, b)] = class_name
    return color2class

def get_class_to_cityscapes_id_conversion_table():
    # Create a dictionary converting KITTI class strings to Cityscapes IDs
    # Cityscapes IDs, for reference:
    """
    Cityscapes IDs, for reference:
    0: road
    1: sidewalk
    2: building
    3: wall
    4: fence
    5: pole
    6: traffic light
    7: traffic sign
    8: vegetation
    9: terrain
    10: sky
    11: person
    12: rider
    13: car
    14: truck
    15: bus
    16: train
    17: motorcycle
    18: bicycle
    """
    class2cityscapes_id = {
        "Terrain": 9,
        "Sky": 10,
        "Tree": 8,
        "Vegetation": 8,
        "Building": 2,
        "Pole": 5,
        "TrafficLight": 6,
        "TrafficSign": 7,
        "Road": 0,
        "GuardRail": 4,
        "Car": 13,
        "Truck": 14,
        "Van": 13,
        "Undefined": 255,
        "Misc": 255,
    }
    return class2cityscapes_id

def get_cityscapes_id_to_cityscapes_name_conversion_table():
    # Create a dictionary converting Cityscapes IDs to Cityscapes class names
    cityscapes_id2name = {
        0: "road",
        1: "sidewalk",
        2: "building",
        3: "wall",
        4: "fence",
        5: "pole",
        6: "traffic light",
        7: "traffic sign",
        8: "vegetation",
        9: "terrain",
        10: "sky",
        11: "person",
        12: "rider",
        13: "car",
        14: "truck",
        15: "bus",
        16: "train",
        17: "motorcycle",
        18: "bicycle"
    }
    return cityscapes_id2name

def convert_kitt_mask_to_cityscapes_mask(kitti_mask, color_to_class, class_to_cityscapes_id):
    # Create an empty Cityscapes mask with the same height and width as the KITTI mask
    cityscapes_mask = np.full(kitti_mask.shape[:2], 255, dtype=np.uint8)  # Default to 255 (undefined)

    # OpenCV reads color images as BGR; convert once so lookups match RGB tuples from colors.txt.
    kitti_mask_rgb = cv2.cvtColor(kitti_mask, cv2.COLOR_BGR2RGB)

    # Iterate over each pixel in the KITTI mask
    for y in range(kitti_mask_rgb.shape[0]):
        for x in range(kitti_mask_rgb.shape[1]):
            color = tuple(kitti_mask_rgb[y, x])  # Get the RGB color at this pixel
            class_name = color_to_class.get(color, "Undefined")  # Get the corresponding class name
            cityscapes_id = class_to_cityscapes_id.get(class_name, 255)  # Get the corresponding Cityscapes ID
            # Write the actual Cityscapes label ID (0-18 or 255), not a visualization intensity.
            cityscapes_mask[y, x] = np.uint8(cityscapes_id)

    return cityscapes_mask

def main():
    # Convert every PNG in masks_kitti to Cityscapes format in masks_cityscapes.
    base_dir = Path(__file__).resolve().parent
    kitti_masks_dir = base_dir / "masks_kitti"
    cityscapes_masks_dir = base_dir / "masks_cityscapes"

    color_to_class = get_color_to_class_conversion_table(base_dir / "colors.txt")
    class_to_cityscapes_id = get_class_to_cityscapes_id_conversion_table()

    cityscapes_masks_dir.mkdir(parents=True, exist_ok=True)

    kitti_mask_paths = sorted(kitti_masks_dir.glob("*.png"))
    if not kitti_mask_paths:
        raise FileNotFoundError(f"No PNG files found in: {kitti_masks_dir}")

    for kitti_mask_path in kitti_mask_paths:
        cityscapes_mask_path = cityscapes_masks_dir / kitti_mask_path.name

        kitti_mask = cv2.imread(str(kitti_mask_path), cv2.IMREAD_COLOR)
        if kitti_mask is None:
            print(f"Skipping unreadable mask: {kitti_mask_path}")
            continue

        cityscapes_mask = convert_kitt_mask_to_cityscapes_mask(kitti_mask, color_to_class, class_to_cityscapes_id)
        cv2.imwrite(str(cityscapes_mask_path), cityscapes_mask)
        print(f"Converted: {kitti_mask_path.name} -> {cityscapes_mask_path.name}")

if __name__ == "__main__":
    main()