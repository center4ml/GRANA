from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


transforms = [
    A.Normalize(**{'mean': 0.2845, 'std': 0.1447}, max_pixel_value=1.0),
    # Applies the formula (img - mean * max_pixel_value) / (std * max_pixel_value)
    ToTensorV2()
]

model_config = {
    'receptive_field_height': 220,
    'receptive_field_width': 38,
    'stride_height': 64,
    'stride_width': 2,
    'image_height': 476,
    'image_width': 476}

