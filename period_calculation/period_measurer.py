import torch
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import skimage
import scipy
import numpy as np
from pytorch_lightning import seed_everything

from period_calculation.data_reader import AdHocDataset3
from period_calculation.models.gauss_model import GaussPeriodModel


transforms = [
    A.Normalize(**{'mean': 0.2845, 'std': 0.1447}, max_pixel_value=1.0),
    # Applies the formula (img - mean * max_pixel_value) / (std * max_pixel_value)
    ToTensorV2()
]

class PeriodMeasurer:
    """returns period in pixels"""
    def __init__(
        self, weights_file, image_height=476, image_width=476,
        px_per_nm = 1,
        sd_threshold_nm=np.inf,
        period_threshold_nm_min=0, period_threshold_nm_max=np.inf):
        
        self.model = GaussPeriodModel.load_from_checkpoint(weights_file).to("cpu") #.eval()?
        self.px_per_nm = px_per_nm
        self.sd_threshold_nm = sd_threshold_nm
        self.period_threshold_nm_min = period_threshold_nm_min
        self.period_threshold_nm_max = period_threshold_nm_max
        
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> float:
        seed_everything(44)
        dataset = AdHocDataset3(
            images_and_masks = [(img, mask)],
            transform_level=-1,
            retain_raw_images=False,
            transforms=transforms
        )
        
        image_data = dataset[0]
        with torch.no_grad():
            y_hat, sd_hat = self.model(image_data["image"].unsqueeze(0), return_raw=False)
        
        y_hat_nm = (y_hat/image_data["scale"]).item() / self.px_per_nm
        sd_hat_nm = (sd_hat/image_data["scale"]).item() /self.px_per_nm
        
        
        if (sd_hat_nm>self.sd_threshold_nm) or (y_hat_nm<self.period_threshold_nm_min) or (y_hat_nm>self.period_threshold_nm_max):
            y_hat_nm = np.nan
            
        return y_hat_nm, sd_hat_nm
