from PIL import Image
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
from torchvision.transforms import functional as tvf

from pathlib import Path

def sliced_mean(x, slice_size):
    cs_y = np.cumsum(x, axis=0)
    cs_y = np.concatenate((np.zeros((1, cs_y.shape[1]), dtype=cs_y.dtype), cs_y), axis=0)
    slices_y = (cs_y[slice_size:] - cs_y[:-slice_size])/slice_size
    cs_xy = np.cumsum(slices_y, axis=1)
    cs_xy = np.concatenate((np.zeros((cs_xy.shape[0], 1), dtype=cs_xy.dtype), cs_xy), axis=1)
    slices_xy = (cs_xy[:,slice_size:] - cs_xy[:,:-slice_size])/slice_size
    return slices_xy

def sliced_var(x, slice_size):
    x = x.astype('float64')
    return sliced_mean(x**2, slice_size) - sliced_mean(x, slice_size)**2

def calculate_local_variance(img, var_window):
    """return local variance map with the same size as input image"""
    var = sliced_var(img, var_window)

    left_pad = var_window // 2 -1
    right_pad = var_window -1 - left_pad
    var_padded = np.pad(
        var,
        pad_width=(
            (left_pad,right_pad),
            (left_pad,right_pad)
        ))
    return var_padded

def get_crop_batch(img: np.ndarray, mask: np.ndarray, crop_size=96, crop_scales=np.geomspace(0.5, 2, 7), samples_per_scale=32, use_variance_threshold=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a batch of cropped images from an input image and corresponding mask, at various scales and rotations.

    Parameters
    ----------
    img : np.ndarray
        The input image from which crops are generated.
    mask : np.ndarray
        The binary mask indicating the region of interest in the image.
    crop_size : int, optional
        The size of the square crop.
    crop_scales : np.ndarray, optional
        An array of scale factors to apply to the crop size.
    samples_per_scale : int, optional
        Number of samples to generate per scale factor.
    use_variance_threshold : bool, optional
        Flag to use variance thresholding for selecting crop locations.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing the tensor of crops, their rotation angles, and scale factors.
    """
    
    # pad
    pad_size = int(np.ceil(0.5*crop_size*max(crop_scales)*(np.sqrt(2)-1)))
    img_padded = np.pad(img, pad_size)
    mask_padded = np.pad(mask, pad_size)

    # distance map
    distance_map_padded = ndimage.distance_transform_edt(mask_padded)
    # TODO: adjust scales and samples_per_scale
    
    if use_variance_threshold:
        variance_window = min(crop_size//2, min(img.shape))
        variance_map_padded = np.pad(calculate_local_variance(img, variance_window), pad_size)
        variance_median = np.ma.median(np.ma.masked_where(distance_map_padded<0.5*variance_window, variance_map_padded))
        variance_mask = variance_map_padded >= variance_median
    else:
        variance_mask = np.ones_like(mask_padded)
    
    # initilize output
    crops_granum = []
    angles_granum = []
    scales_granum = []
    # loop over scales
    for scale in crop_scales: 
        half_crop_size_scaled = int(np.floor(scale*0.5*crop_size)) # half of crop size after scaling
        crop_pad = int(np.ceil((np.sqrt(2) - 1)*half_crop_size_scaled)) # pad added in order to allow rotation
        half_crop_size_external = half_crop_size_scaled + crop_pad # size of "external crop" which will be rotated

        possible_indices = np.stack(np.where(variance_mask & (distance_map_padded >= 2*half_crop_size_scaled)), axis=1)
        if len(possible_indices) == 0:
            continue
        chosen_indices = np.random.choice(np.arange(len(possible_indices)), min(len(possible_indices), samples_per_scale), replace=False)

        crops = [
            img_padded[y-half_crop_size_external:y+half_crop_size_external, x-half_crop_size_external:x+half_crop_size_external] for y, x in possible_indices[chosen_indices]
        ]

        # rotate
        rotation_angles = np.random.rand(len(crops))*180 - 90
        crops = [
            ndimage.rotate(crop, angle, reshape=False)[crop_pad:-crop_pad,crop_pad:-crop_pad] for crop, angle in zip(crops, rotation_angles)
        ]
        # add to output
        crops_granum.append(tvf.resize(torch.tensor(np.array(crops)), (crop_size,crop_size),antialias=True)) # resize crops to crop_size
        angles_granum.extend(rotation_angles.tolist())
        scales_granum.extend([scale]*len(crops))

    if len(angles_granum) == 0:
        return [], [], []
    
    crops_granum = torch.concat(crops_granum)
    angles_granum = torch.tensor(angles_granum, dtype=torch.float)
    scales_granum = torch.tensor(scales_granum, dtype=torch.float)
    
    return crops_granum, angles_granum, scales_granum

def get_crop_batch_from_path(img_path, mask_path=None, use_variance_threshold=False):
    """
    Load an image and its mask from file paths and generate a batch of cropped images.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    mask_path : str, optional
        Path to the binary mask image. If None, assumes mask path by replacing image extension with '.npy'.
    use_variance_threshold : bool, optional
        Flag to use variance thresholding for selecting crop locations.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing the tensor of crops, their rotation angles, and scale factors, obtained from the specified image path.
    """
    if mask_path is None:
        mask_path = str(Path(img_path).with_suffix('.npy'))
    mask = np.load(mask_path)
    img = np.array(Image.open(img_path))[:,:,0]
    
    return get_crop_batch(img, mask, use_variance_threshold=use_variance_threshold)
    