from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import ndimage

from dataclasses import dataclass
from typing import Any, List
from zipfile import ZipFile

def add_text(image: Image.Image, text: str, location=(0.5, 0.5), color='red', size=40) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=size)
    draw.text((int(image.size[0]*location[0]), int(image.size[1]*location[1])), text, font=font, fill=color)
    return image


def select_unique_mask(mask):
    """if mask consists of multiple parts, select the largest"""
    blobs = ndimage.label(mask)[0]
    blob_labels, blob_sizes = np.unique(blobs, return_counts=True)
    best_blob_label = blob_labels[1:][np.argmax(blob_sizes[1:])]
    return blobs == best_blob_label

def object_slice(mask, margin=128):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Create a slice object for the bounding box
    bounding_box_slice = (
        slice(max(0,row_min-margin), min(row_max + 1+margin, len(rows)+1)),
        slice(max(0,col_min-margin), min(col_max + 1+margin, len(cols)+1))
    )
    
    return bounding_box_slice

def resize_to(image: Image.Image, s=4032) -> Image.Image:
    w, h = image.size
    longest_size = max(h, w)
        
    resize_factor = longest_size / s
        
    resized_image = image.resize((int(w/resize_factor), int(h/resize_factor)))
    return resized_image

def rolling_mean(x, window):
    cs = np.r_[0, np.cumsum(x)]
    rolling_sum = cs[window:] - cs[:-window]
    return rolling_sum/window

@dataclass
class Granum:
    image: Any = None#Optional[np.ndarray] = None
    mask: Any = None #Optional[np.ndarray] = None
    scaler: Any = None
    nm_per_px: float = float('nan')
    detection_confidence: float = float('nan')

def zip_files(files: List[str], output_name: str) -> None:
    with ZipFile(output_name, "w") as zipObj:
        for file in files:
            zipObj.write(file)

def filter_boundary_detections(masks, scaler=None):
    last_index_right = -1 if scaler is None else masks.shape[1]-1-scaler.pad_right
    last_index_bottom = -1 if scaler is None else masks.shape[2]-1-scaler.pad_bottom
    doesnt_touch_boundary_mask = ~(np.any(masks[:,0,:] != 0, axis=1) | np.any(masks[:,last_index_right:,:] != 0, axis=(1,2)) | np.any(masks[:,:,0] != 0, axis=1) | np.any(masks[:,:,last_index_bottom:] != 0, axis=(1,2)))
    return doesnt_touch_boundary_mask

def get_circle_mask(shape, r=None):
    if isinstance(shape, int):
        shape = (shape, shape)
    if r is None:
        r = min(shape)/2
    X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    center_x = shape[1] / 2 - 0.5
    center_y = shape[0] / 2 - 0.5

    mask = ((X-center_x)**2 + (Y-center_y)**2) >= r**2
    return mask