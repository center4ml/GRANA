import itertools
import warnings
from io import BytesIO
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure
import torchvision.transforms.functional as tvf
from ultralytics import YOLO
import torch
import cv2
import gradio

import sys, os
sys.path.append(os.path.abspath('angle_calculation'))
# from classic import measure_object
from sampling import get_crop_batch
from angle_model import PatchedPredictor, StripsModelLumenWidth

from period_calculation.period_measurer import PeriodMeasurer
# from grana_detection.mmwrapper import MMDetector # mmdet installation in docker is problematic for now

@dataclass
class Granum:
    id: Optional[int] = None
    image: Any = None
    mask: Any = None
    scaler: Any = None
    nm_per_px: float = float('nan')
    detection_confidence: float = float('nan')
    img_oriented: Optional[np.ndarray] = None # oriented fragment of the image
    mask_oriented: Optional[np.ndarray] = None # oriented fragment of the mask
    measurements: dict = field(default_factory=dict) # dict with grana measurements

class ScalerPadder:
    """resize and pad image to specific range.
    minimal_pad: obligatory padding, e.g. required for detector
    """
    def __init__(self, target_size=1024, target_short_edge_min=640, minimal_pad=16, pad_to_multiply=32):
        self.minimal_pad = minimal_pad
        self.target_size = target_size - 2*self.minimal_pad # detection pad is necessary padding size
        self.target_short_edge_min = target_short_edge_min - 2*self.minimal_pad

        self.max_size_nm = 6000 # training images covers ~3100 nm
        self.min_size_nm = 2400 # training images covers ~3100 nm
        self.pad_to_multiply = pad_to_multiply
        
    def transform(self, image: Image.Image, px_per_nm: float=1.298) -> Image.Image:
        self.original_size = image.size
        self.original_px_per_nm = px_per_nm
        w, h = self.original_size
        longest_size = max(h, w)
        img_size_nm = longest_size / px_per_nm
        if img_size_nm > self.max_size_nm:
            error_message = f'too large image, image size: {img_size_nm:0.1f}nm, max allowed: {self.max_size_nm}nm'
            # raise ValueError(error_message)
            # warnings.warn(warning_message)
            gradio.Warning(error_message)
            # add_text(image, warning_message, location=(0.1, 0.1), color='blue', size=int(40*longest_size/self.target_size))
        
        self.resize_factor = self.target_size / (max(self.min_size_nm, img_size_nm) * px_per_nm)
        self.px_per_nm_transformed = px_per_nm * self.resize_factor
        
        resized_image = resize_with_cv2(image, (int(h*self.resize_factor), int(w*self.resize_factor)))
        
        if w >= h:
            pad_w = self.target_size-resized_image.size[0]
            pad_h = max(0, self.target_short_edge_min-resized_image.size[1])
        else:
            pad_w = max(0, self.target_short_edge_min-resized_image.size[0])
            pad_h = self.target_size-resized_image.size[1]
        
        # apply minimal padding
        pad_w += 2*self.minimal_pad
        pad_h += 2*self.minimal_pad
        
        # round to multiplication
        pad_w += (self.pad_to_multiply - resized_image.size[0]%self.pad_to_multiply)%self.pad_to_multiply
        pad_h += (self.pad_to_multiply - resized_image.size[1]%self.pad_to_multiply)%self.pad_to_multiply
        
        self.pad_right = pad_w // 2
        self.pad_left = pad_w - self.pad_right
        
        self.pad_up = pad_h // 2
        self.pad_bottom = pad_h - self.pad_up
        
        padded_image = tvf.pad(resized_image, [self.pad_left,self.pad_up, self.pad_right, self.pad_bottom], padding_mode='reflect') # fill 114 as in YOLO
        return padded_image
    
    @property
    def unpad_slice(self) -> Tuple[slice]:
        return slice(self.pad_up,-self.pad_bottom if self.pad_bottom>0 else None), slice(self.pad_left,-self.pad_right if self.pad_right>0 else None)
    
    def inverse_transform(self, image: Union[np.ndarray, Image.Image], output_size: Optional[Tuple[int]]=None, output_nm_per_px: Optional[float]=None, return_pil: bool=True) -> Image.Image:
        if isinstance(image, Image.Image):
            image = np.array(image)
        # h, w = image.shape[:2]
        # unpadded_image = image[self.pad_up:h-self.pad_bottom,self.pad_left:w-self.pad_right]
        unapdded_image = image[self.unpad_slice]
        
        if output_size is not None and output_nm_per_px is not None:
            raise ValueError("one of output_size or output_nm_per_px must not be None")
        elif output_nm_per_px is not None:
            resize_factor = self.original_nm_per_px/output_nm_per_px
            output_size = (int(self.original_size[0]*resize_factor), int(self.original_size[1]*resize_factor))
        elif output_size is None:
            output_size = self.original_size
        resized_image = resize_with_cv2(unapdded_image, (output_size[1],output_size[0]), return_pil=return_pil) #Image.fromarray(unpadded_image).resize(self.original_size)
        
        return resized_image
    
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tol=0.01):
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    # assert len(contours) == 1 #raise error if there are more than 1 contour
    contour = contours[0]
    contour -= 1 # correct for padding
    contour = close_contour(contour)

    polygon = measure.approximate_polygon(contour, tol)

    polygon = np.flip(polygon, axis=1)
    # after padding and subtracting 1 we may get -0.5 points in our polygon. Replace it with 0
    polygon = np.where(polygon>=0, polygon, 0)
    # segmentation = polygon.ravel().tolist()

    return polygon

def measure_shape(binary_mask):
    contour = binary_mask_to_polygon(binary_mask)
    perimeter = np.sum(np.linalg.norm(contour[:-1] - contour[1:], axis=1))
    area = np.sum(binary_mask)
        
    return perimeter, area

def calculate_gsi(perimeter, height, area):
    a = 0.5*(perimeter - 2*height)
    return 1 - area/(a*height)

def object_slice(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Create a slice object for the bounding box
    bounding_box_slice = (slice(row_min, row_max + 1), slice(col_min, col_max + 1))
    
    return bounding_box_slice


def figure_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Load the image from the buffer as a PIL Image
    image = deepcopy(Image.open(buf))

    # Close the buffer
    buf.close()
    return image

def resize_to(image: Image.Image, s: int=4032, return_factor: bool =False) -> Image.Image:
    w, h = image.size
    longest_size = max(h, w)
        
    resize_factor = longest_size / s
        
    resized_image = image.resize((int(w/resize_factor), int(h/resize_factor)))
    if return_factor:
        return resized_image, resize_factor
    return resized_image

def resize_with_cv2(image, shape, return_pil=True):
    """resize using cv2 with cv2.INTER_LINEAR - consistent with YOLO"""
    h, w = shape
    if isinstance(image, Image.Image):
        image = np.array(image)
        
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    if return_pil:
        return Image.fromarray(resized)
    else:
        return resized

def select_unique_mask(mask):
    """if mask consists of multiple parts, select the largest"""
    if not np.any(mask): # if mask is empty, return without change
        return mask
    blobs = ndimage.label(mask)[0]
    blob_labels, blob_sizes = np.unique(blobs, return_counts=True)
    best_blob_label = blob_labels[1:][np.argmax(blob_sizes[1:])]
    return blobs == best_blob_label

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

def calculate_distance_map(mask):
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
    distance_map_padded = ndimage.distance_transform_edt(padded)
    return distance_map_padded[1:-1,1:-1]

def select_samples(granum_image, granum_mask, crop_size=96, n_samples=64, granum_fraction_min=0.75, variance_p=0.):
    granum_occupancy = sliced_mean(granum_mask, crop_size)
    possible_indices = np.stack(np.where(granum_occupancy >= granum_fraction_min), axis=1)
    
    if variance_p == 0:
        p = np.ones(len(possible_indices))
    else:
        variance_map = sliced_var(granum_image, crop_size)
        p = variance_map[possible_indices[:,0], possible_indices[:,1]]**variance_p
    p /= np.sum(p)
    
    chosen_indices = np.random.choice(
        np.arange(len(possible_indices)),
        min(len(possible_indices), n_samples),
        replace=False,
        p = p
    )

    crops = []
    for crop_idx, idx in enumerate(chosen_indices):
        crops.append(
            granum_image[
                possible_indices[idx,0]:possible_indices[idx,0]+crop_size,
                possible_indices[idx,1]:possible_indices[idx,1]+crop_size
            ]
        )
    return np.array(crops)

def calculate_height(mask_oriented): #HACK
    span = mask_oriented.shape[0] - np.argmax(mask_oriented[::-1], axis=0) - np.argmax(mask_oriented, axis=0)
    return np.quantile(span, 0.8) 

def calculate_diameter(mask_oriented):
    """returns mean diameter"""
    # calculate 0.25 and 0.75 lines
    vertical_mask = np.any(mask_oriented, axis=1)
    upper_granum_bound = np.argmax(vertical_mask)
    lower_granum_bound = mask_oriented.shape[0] - np.argmax(vertical_mask[::-1])
    upper = round(0.75*upper_granum_bound + 0.25*lower_granum_bound)
    lower = max(upper+1, round(0.25*upper_granum_bound + 0.75*lower_granum_bound))
    valid_rows_slice = slice(upper, lower)
    
    # calculate diameters
    span = mask_oriented.shape[1] - np.argmax(mask_oriented[valid_rows_slice,::-1], axis=1) - np.argmax(mask_oriented[valid_rows_slice], axis=1)
    return np.mean(span)

def robust_mean(x, q=0.1):
    x_med = np.median(x)
    deviations = abs(x- x_med)
    if max(deviations) == 0:
        mask = np.ones(len(x), dtype='bool')
    else:
        threshold = np.quantile(deviations, 1-q)
        mask = x[deviations<= threshold]
     
    return np.mean(x[mask])

def rotate_image_and_mask(image, mask, direction):
    mask_oriented = ndimage.rotate(mask.astype('int'), -direction, reshape=True).astype('bool')
    idx_begin_x, idx_end_x = np.where(np.any(mask_oriented, axis=0))[0][np.array([0, -1])]
    idx_begin_y, idx_end_y = np.where(np.any(mask_oriented, axis=1))[0][np.array([0, -1])]
    img_oriented = ndimage.rotate(image, -direction, reshape=True) #[idx_begin_y:idx_end_y, idx_begin_x:idx_end_x]
    return img_oriented, mask_oriented

class GranaAnalyser:
    def __init__(self, weights_detector: str, weights_orientation: str, weights_period: str, period_sd_threshold_nm: float=2.5) -> None:
        """
        Initializes the GranaAnalyser with specified weights for detection and measuring.

        This method loads the weights for the grana detection and measuring algorithms 
        from the specified file paths. It also loads mock data for visualization and 
        analysis purposes.

        Parameters:
        weights_detector (str): The file path to the weights file for the grana detection algorithm.
        weights_orientation (str): The file path to the weights file for the grana orientation algorithm.
        weights_period (str): The file path to the weights file for the grana period algorithm.
        """
        self.detector = YOLO(weights_detector)
        
        self.orienter = PatchedPredictor(
            StripsModelLumenWidth.load_from_checkpoint(weights_orientation, map_location='cpu').eval(),
            normalization = dict(mean=0.250, std=0.135),
            n_samples=32,
            mask=None,
            crop_size=64,
            angle_confidence_threshold=0.2
        )

        self.measurement_px_per_nm = 1/0.768 # image scale required for measurement

        self.period_measurer = PeriodMeasurer(
            weights_period,
            px_per_nm=self.measurement_px_per_nm,
            sd_threshold_nm=period_sd_threshold_nm,
            period_threshold_nm_min=14, period_threshold_nm_max=30
        )
        

    def get_grana_data(self, image, detections, scaler, border_margin=1, min_count=1) -> List[Granum]:
        """filter detections and create grana data"""
        image_numpy = np.array(image)
        if image_numpy.ndim == 3:
            image_numpy = image_numpy[:,:,0]
        
        mask_all = None
        grana = []
        for mask, confidence in zip(
            detections.masks.data.cpu().numpy().astype('bool'),
            detections.boxes.conf.cpu().numpy()
        ):
            granum_mask = select_unique_mask(mask[scaler.unpad_slice])
            # check if mask is empty after padding
            if not np.any(granum_mask):
                continue
            granum_mask = ndimage.binary_fill_holes(granum_mask)

            # check if touches boundary:
            if (np.sum(granum_mask[:border_margin])>min_count) or \
                (np.sum(granum_mask[-border_margin:])>min_count) or \
                (np.sum(granum_mask[:,:border_margin])>min_count) or \
                (np.sum(granum_mask[:,-border_margin:])>min_count):
                
                continue
            
            # check grana overlap
            if mask_all is None:
                mask_all = granum_mask
            else:
                intersection = mask_all & granum_mask
                
                if intersection.sum() >= (granum_mask.sum() * 0.2):
                    continue
                mask_all = mask_all | granum_mask
            
            granum = Granum(
                image = image,
                mask = granum_mask,
                scaler=scaler,
                detection_confidence=float(confidence)
            )
            
            granum.image_numpy = image_numpy
            grana.append(granum)
        return grana
    
    def measure_grana(self, grana: List[Granum], measurement_image: np.ndarray) -> List[Granum]:
        """measure grana: includes orientation detection, period detection and geometric measurements"""
        for granum in grana:
            measurement_mask = resize_with_cv2(granum.mask.astype(np.uint8), measurement_image.shape[:2], return_pil=False).astype('bool')
            
            granum.bounding_box_slice = object_slice(measurement_mask)
            granum.image_crop = measurement_image[granum.bounding_box_slice][:,:]
            granum.mask_crop = measurement_mask[granum.bounding_box_slice]
                        
            # initialize measurements
            granum.measurements = {}

            # measure shape
            granum.measurements['perimeter px'], granum.measurements['area px'] = measure_shape(granum.mask_crop)

            # measrure orientation
            orienter_predictions = self.orienter(granum.image_crop, granum.mask_crop)
            granum.measurements['direction'] = orienter_predictions["est_angle"]
            granum.measurements['direction confidence'] = orienter_predictions["est_angle_confidence"]
            
            if not np.isnan(granum.measurements["direction"]):
                img_oriented, mask_oriented = rotate_image_and_mask(granum.image_crop, granum.mask_crop, granum.measurements["direction"])
                oriented_granum_slice = object_slice(mask_oriented)
                granum.img_oriented = img_oriented[oriented_granum_slice]
                granum.mask_oriented = mask_oriented[oriented_granum_slice]
                granum.measurements['height px'] = calculate_height(granum.mask_oriented)
                granum.measurements['GSI'] = calculate_gsi(
                    granum.measurements['perimeter px'],
                    granum.measurements['height px'],
                    granum.measurements['area px']
                )
                granum.measurements['diameter px'] = calculate_diameter(granum.mask_oriented)
                                
                oriented_granum_slice = object_slice(granum.mask_oriented)
                granum.measurements["period nm"], granum.measurements["period SD nm"] = self.period_measurer(granum.img_oriented, granum.mask_oriented)
                
                if not pd.isna(granum.measurements['period nm']):
                    granum.measurements['Number of layers'] = round(granum.measurements['height px']/ self.measurement_px_per_nm / granum.measurements['period nm'])
            
        return grana

    def extract_grana_data(self, grana: List[Granum]) -> pd.DataFrame:
        """collect and scale grana data"""
        grana_data = []
        for granum in grana:
            granum_entry = {
                'Granum ID': granum.id,
                'detection confidence': granum.detection_confidence
            }
            # fill with None if absent:
            for key in ['direction', 'Number of layers', 'GSI', 'period nm', 'period SD nm']:
                granum_entry[key] = granum.measurements.get(key, None)
            # scale linearly:
            for key in ['height px', 'diameter px', 'perimeter px', 'perimeter px']:
                granum_entry[f"{key[:-3]} nm"] = granum.measurements.get(key, np.nan) / self.measurement_px_per_nm
            # scale quadratically
            granum_entry['area nm^2'] = granum.measurements['area px'] / self.measurement_px_per_nm**2
            
            grana_data.append(granum_entry)
                
        return pd.DataFrame(grana_data)
    
    def visualize_detections(self, grana: List[Granum], image: Image.Image) -> Image.Image:
        visualization_longer_edge = 1024
        scale = visualization_longer_edge/max(image.size)
        visualization_size = (round(scale*image.size[0]), round(scale*image.size[1]))
        visualization_image = np.array(image.resize(visualization_size).convert('RGB'))
        
        if len(grana) > 0:
            grana_mask = resize_with_cv2(
                np.any(np.array([granum.mask for granum in grana]),axis=0).astype(np.uint8),
                visualization_size[::-1],
                return_pil=False
            ).astype('bool')
            visualization_image[grana_mask]= (0.7*visualization_image[grana_mask] + 0.3*np.array([[[39, 179, 115]]])).astype(np.uint8)

        
        for granum in grana:
            scale = visualization_longer_edge/max(granum.mask.shape)
            y, x = ndimage.center_of_mass(granum.mask)
            cv2.putText(visualization_image, f'{granum.id}', org=(int(x*scale)-10, int(y*scale)+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX , fontScale=1, color=(39, 179, 115),thickness = 2)
        
        return Image.fromarray(visualization_image)       
        

    def generate_grana_images(self, grana: List[Granum], image_name: str ="") -> List[Image.Image]:
        grana_images = {}
        for granum in grana:
            fig, ax = plt.subplots()
            if granum.img_oriented is None:
                image_to_plot = granum.image_crop
                mask_to_plot = granum.mask_crop
                extra_caption = " orientation and period unknown"
            else:
                image_to_plot = granum.img_oriented
                mask_to_plot = granum.mask_oriented
                extra_caption = ""
            
            ax.imshow(0.5*255*(~mask_to_plot) +image_to_plot*(1-0.5*(~mask_to_plot)), cmap='gray', vmin=0, vmax=255)
            ax.axis('off')
            ax.set_title(f'[{granum.id}]{image_name}\n{extra_caption}')
            granum_image = figure_to_pil(fig)
            grana_images[granum.id] = granum_image
            plt.close('all')
        
        return grana_images

    def format_data(self, grana_data: pd.DataFrame) -> pd.DataFrame:
        rounding_roles = {'area nm^2': 0, 'perimeter nm': 1, 'diameter nm': 1, 'height nm': 1, 'period nm': 1, 'period SD nm': 2, 'GSI':2, 'direction': 1}
        rounded_data = grana_data.round(rounding_roles)
        columns_order = ['Granum ID', 'File name', 'area nm^2', 'perimeter nm', 'GSI','diameter nm', 'height nm', 'Number of layers','period nm', 'period SD nm', 'direction']
        return rounded_data[columns_order]
        
    
    def aggregate_data(self, grana_data: pd.DataFrame, confidence: Optional[float]=None) -> Dict:
        if confidence is None:
            filtered = grana_data
        else:
            filtered = grana_data.loc[grana_data['aggregated confidence'] >=  confidence]
        aggregation = filtered[['area nm^2', 'perimeter nm', 'diameter nm', 'height nm', 'Number of layers', 'period nm', 'GSI']].mean().to_dict()
        aggregation_std = filtered[['area nm^2', 'perimeter nm', 'diameter nm', 'height nm', 'Number of layers', 'period nm', 'GSI']].std().to_dict()
        aggregation_std = {f"{k} std": v for k, v in aggregation_std.items()}
        aggregation_result = {**aggregation, **aggregation_std, 'N grana': len(filtered)}
        return aggregation_result
            
    def predict_on_single(self, image: Image.Image, scale: float, detection_confidence: float=0.25, granum_id_start=1, image_name: str = "") -> Tuple[List[Image.Image], pd.DataFrame, List[Image.Image]]:
        """
        Predicts and aggregates data related to grana using a dictionary of images.

        Parameters:
        image (Image.Image): PIL Image object to be analyzed
        scale (float): scale of the image: px per nm.
        detection_confidence (float): The detection confidence threshold shape measurement

        Returns:
        Tuple[Image.Image, pandas.DataFrame, List[Image.Image]]: 
        A tuple containing:
               - detection_visualization (Image.Image): PIL image representing 
                 the detection visualizations.
               - grana_data (pandas.DataFrame): A DataFrame containing the simulated granum data.
               - grana_images (List[Image.Image]): A list of PIL images of the grana.
        """
        # convert to grayscale
        image = image.convert("L")
                
        # detect
        scaler = ScalerPadder(target_size=1024, target_short_edge_min=640)
        scaled_image = scaler.transform(image, px_per_nm=scale)
        detections = self.detector.predict(source=scaled_image, conf=detection_confidence)[0]

        # get grana data
        grana = self.get_grana_data(image, detections, scaler)
        for granum_id, granum in enumerate(grana, start=granum_id_start):
            granum.id = granum_id

        
        # visualize detections
        detection_visualization = self.visualize_detections(grana, image)
        
        # measure grana
        measurement_image_resize_factor = self.measurement_px_per_nm / scale
        measurement_image_shape = (
            int(image.size[1]*measurement_image_resize_factor),
            int(image.size[0]*measurement_image_resize_factor)
        )
        measurement_image = resize_with_cv2(  # numpy image in scale valid for measurement
            image, measurement_image_shape, return_pil=False
        )
        grana = self.measure_grana(grana, measurement_image)

        # pandas DataFrame
        grana_data = self.extract_grana_data(grana)
        
        # list of PIL images
        grana_images = self.generate_grana_images(grana, image_name=image_name)
        
        return detection_visualization, grana_data, grana_images

    def predict(self, images: Dict[str, Image.Image], scale: float, detection_confidence: float=0.25, parameter_confidence: Optional[float]=None) -> Tuple[List[Image.Image], pd.DataFrame, List[Image.Image], Dict]:
        """
        Predicts and aggregates data related to grana using a dictionary of images.

        Parameters:
        images (Dict[str, Image.Image]): A dictionary of PIL Image objects to be analyzed, 
                                         keyed by their names.
        scale (float): scale of the image: px per nm
        detection_confidence (float): The detection confidence threshold shape measurement
        parameter_confidence (float): The confidence threshold used for data aggregation. Only 
                            data with aggregated confidence above this threshold will 
                            be considered.

        Returns:
        Tuple[List[Image.Image], pandas.DataFrame, List[Image.Image], Dict]: 
        A tuple containing:
               - detection_visualizations (List[Image.Image]): A list of PIL images representing 
                 the detection visualizations.
               - grana_data (pandas.DataFrame): A DataFrame containing the simulated granum data.
               - grana_images (List[Image.Image]): A list of PIL images of the grana.
               - aggregated_data (Dict): A dictionary containing the aggregated data results.
        """
        detection_visualizations_all = {}
        grana_data_all = None
        grana_images_all = {}
        
        granum_id_start = 1
        for image_name, image in images.items():
            detection_visualization, grana_data, grana_images = self.predict_on_single(image, scale=scale, detection_confidence=detection_confidence, granum_id_start=granum_id_start, image_name=image_name)
            granum_id_start += len(grana_data)
            detection_visualizations_all[image_name] = detection_visualization
            grana_images_all.update(grana_images)
            
            grana_data['File name'] = image_name
            if grana_data_all is None:
                grana_data_all = grana_data
            else:
                # grana_data['Granum ID'] += len(grana_data_all)
                grana_data_all = pd.concat([grana_data_all, grana_data])
            
        # dict
        # grana_data_all.to_csv('grana_data_all.csv', index=False)
        aggregated_data = self.aggregate_data(grana_data_all, parameter_confidence)
        
        formatted_grana_data = self.format_data(grana_data_all)
        
        return detection_visualizations_all, formatted_grana_data, grana_images_all, aggregated_data

    
class GranaDetector(GranaAnalyser):
    """supplementary class for grana detection only
    """
    def __init__(self, weights_detector: str, detector_config: Optional[str] = None, model_type="yolo") -> None:

        if model_type == "yolo":
            self.detector = YOLO(weights_detector)
        elif model_type == "mmdetection":
            self.detector = MMDetector(model=detector_config, weights=weights_detector)
        else:
            raise NotImplementedError()
        
    def predict_on_single(self, image: Image.Image, scale: float, detection_confidence: float=0.25, granum_id_start=1, use_scaling=True, granum_border_margin=1, granum_border_min_count=1, scaler_sizes=(1024, 640)) -> List[Granum]:
        # convert to grayscale
        image = image.convert("L")
                
        # detect
        if use_scaling:
            scaler = ScalerPadder(target_size=scaler_sizes[0], target_short_edge_min=scaler_sizes[1])
        else:
            #dummy scaler
            scaler = ScalerPadder(target_size=max(image.size), target_short_edge_min=min(image.size), minimal_pad=0, pad_to_multiply=1)
        scaled_image = scaler.transform(image, scale=scale)
        detections = self.detector.predict(source=scaled_image, conf=detection_confidence)[0]

        # get grana data
        grana = self.get_grana_data(image, detections, scaler, border_margin=granum_border_margin, min_count=granum_border_min_count)
        for i_granum, granum in enumerate(grana, start=1):
            granum.id = i_granum
        
        return grana