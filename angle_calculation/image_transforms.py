import numpy as np
import cv2

def batched_radon(image_batch):
    batch_size, img_size = image_batch.shape[:2]
    if batch_size > 512: # limit batch size to 512 because cv2.warpAffine fails for batch> 512
        return np.concatenate([batched_radon(image_batch[i:i+512]) for i in range(0,batch_size,512)], axis=0)
    theta = np.arange(180)
    radon_image = np.zeros((image_batch.shape[0], img_size, len(theta)),
                           dtype='float32')

    for i, angle in enumerate(theta):
        M = cv2.getRotationMatrix2D(((img_size-1)/2.0,(img_size-1)/2.0),angle,1)
        rotated = cv2.warpAffine(np.transpose(image_batch, (1, 2, 0)),M,(img_size,img_size))
        if batch_size == 1: # cv2.warpAffine cancels batch dimension if equal to 1
          rotated = rotated[:,:, np.newaxis]
        rotated = np.transpose(rotated, (2, 0, 1))
        rotated = rotated / np.array(255, dtype='float32')
        radon_image[:, :, i] = rotated.sum(axis=1)
    return radon_image

def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int):
    """from https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/crops/functional.py"""
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2

def center_crop(img: np.ndarray, crop_height: int, crop_width: int):
    height, width = img.shape[:2]
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[y1:y2, x1:x2]
    return img