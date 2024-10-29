import numpy as np
from scipy import signal
from scipy import ndimage
from scipy.fftpack import next_fast_len
from skimage.transform import rotate
from skimage._shared.utils import convert_to_float
from skimage.transform import warp
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy

def get_directional_std(image, theta=None,*, preserve_range=False):
    
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    image = convert_to_float(image.copy(), preserve_range) #TODO: needed?

    shape_min = min(image.shape)
    img_shape = np.array(image.shape)

    # Crop image to make it square
    slices = tuple(slice(int(np.ceil(excess / 2)),
                      int(np.ceil(excess / 2) + shape_min))
                if excess > 0 else slice(None)
                for excess in (img_shape - shape_min))
    image = image[slices]
    shape_min = min(image.shape)
    img_shape = np.array(image.shape)

    radius = shape_min // 2
    coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                      dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    outside_reconstruction_circle = dist > radius ** 2
    image[outside_reconstruction_circle] = 0

    valid_square_slice = slice(int(np.ceil(radius*(1-1/np.sqrt(2)))), int(np.ceil(radius*(1+1/np.sqrt(2)))) )

    # padded_image is always square
    if image.shape[0] != image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = image.shape[0] // 2
    result = np.zeros(len(theta))

    for i, angle in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(image, R, clip=False)
        result[i] = rotated[valid_square_slice, valid_square_slice].std(axis=0).mean()
    return result

def acf2d(x, nlags=None):
  xo = x - x.mean(axis=0)
  n = len(x)
  if nlags is None:
    nlags = n -1
  lag_len = nlags

  xi = np.arange(1, n + 1)
  d = np.expand_dims(np.hstack((xi, xi[:-1][::-1])),1)

  nobs = len(xo)
  n = next_fast_len(2 * nobs + 1)
  Frf = np.fft.fft(xo, n=n, axis=0)

  acov = np.fft.ifft(Frf * np.conjugate(Frf), axis=0)[:nobs] / d[nobs - 1 :]
  acov = acov.real
  ac = acov[: nlags + 1] / acov[:1]
  return ac

def get_period(acf_table, n_samples=50):
    #TODO: use peak heights to select best candidates. use std to eliminate outliers
    period_candidates = []
    period_candidates_hights = []
    for i in np.random.randint(0, acf_table.shape[1], min(acf_table.shape[1], n_samples)):
        peaks = signal.find_peaks(acf_table[:,i])[0]
        if len(peaks) == 0:
            continue
        peak_idx = peaks[0]
        period_candidates.append(peak_idx)
        period_candidates_hights.append(acf_table[peak_idx,i])
    period_candidates = np.array(period_candidates)
    period_candidates_hights = np.array(period_candidates_hights)
    
    if len(period_candidates) == 0:
        return np.nan, np.nan
    elif len(period_candidates) == 1:
        return period_candidates[0], np.nan
    q1, q3 = np.quantile(period_candidates, [0.25, 0.75])
    candidates_std = np.std(period_candidates[(period_candidates>=q1)&(period_candidates<=q3)])
    # return period_candidates, period_candidates_hights
    return np.median(period_candidates), candidates_std

def get_rotation_with_confidence(padded_image, blur_size=4, make_plots=True):
  std_by_angle = get_directional_std(cv2.blur(padded_image, (blur_size,blur_size)))
  rotation_angle = np.argmin(std_by_angle)

  rotation_quality = 1 - np.min(std_by_angle)/np.median(std_by_angle)
  if make_plots:
    plt.plot(std_by_angle)
    plt.axvline(rotation_angle, c='k')
    plt.title(f'quality: {rotation_quality:0.2f}')
  return rotation_angle, rotation_quality

def calculate_autocorrelation(oriented_img, blur_kernel=(7,1), make_plots=True):
  autocorrelation = acf2d(cv2.blur(oriented_img.T, blur_kernel))
  if make_plots:
    fig, axs = plt.subplots(ncols=2, figsize=(12,6))
    axs[0].imshow(autocorrelation)
    axs[1].plot(autocorrelation.sum(axis=1))
  return autocorrelation

def get_period_with_confidence(autocorrelation_tab, n_samples=30):
    period, period_std = get_period(autocorrelation_tab, n_samples=n_samples)
    if period_std == np.nan:
        period_confidence = 0.001
    else:
        period_confidence = period/(period+2*period_std)
    return period, period_confidence

def calculate_white_fraction(img, blur_size=4, make_plots=True): #TODO: add mask
  blurred = cv2.blur(img, (blur_size, blur_size))
  blurred_sum = blurred.sum(axis=0)
  lower, upper = np.quantile(blurred_sum, [0.15, 0.85])
  sign = blurred_sum > (lower+upper)/2

  sign_change = sign[:-1] != sign[1:]
  sign_change_indices = np.where(sign_change)[0]

  if len(sign_change_indices) >= 2 + (sign[-1] == sign[0]):
    cut_first = sign_change_indices[0]+1

    if sign[-1] == sign[0]:
      cut_last = sign_change_indices[-2]
    else:
      cut_last = sign_change_indices[-1]

    white_fraction = np.mean(sign[cut_first:cut_last])
  else:
    white_fraction = np.nan
    cut_first, cut_last = None, None 
  if make_plots:
      fig, axs = plt.subplots(ncols=3, figsize=(16,6))
      blurred_sum_normalized = blurred_sum - blurred_sum.min()
      blurred_sum_normalized /= blurred_sum_normalized.max()
      axs[0].plot(blurred_sum_normalized)
      axs[0].plot(sign)
      axs[1].plot(blurred_sum_normalized[cut_first:cut_last])
      axs[1].plot(sign[cut_first:cut_last])
      axs[2].imshow(img, cmap='gray')
      for i, idx in enumerate(sign_change_indices):
        plt.axvline(idx, c=['r', 'lime'][i%2])
      fig.suptitle(f'fraction: {white_fraction:0.2f}')

  return white_fraction

def process_img_crop(img, nm_per_px=1, make_plots=False, return_extra=False):

  # image must be square
  assert img.shape[0] == img.shape[1]
  crop_size = img.shape[0]

  # find orientation
  rotation_angle, rotation_quality = get_rotation_with_confidence(img, blur_size=4, make_plots=make_plots)

  # rotate and crop image
  crop_margin = int((1 - 1/np.sqrt(2))*crop_size*0.5)
  oriented_img = rotate(img, -rotation_angle)[2*crop_margin:-crop_margin, crop_margin:-crop_margin]

  # calculate autocorrelation
  autocorrelation = calculate_autocorrelation(oriented_img, blur_kernel=(7,1), make_plots=make_plots)

  # find period
  period, period_confidence = get_period_with_confidence(autocorrelation)
  if make_plots:
    print(f'period: {period}, confidence: {period_confidence}')

  # find white fraction
  white_fraction = calculate_white_fraction(oriented_img, make_plots=make_plots)
  white_width = white_fraction*period
  
  result = {
      'direction': rotation_angle,
      'direction confidence': rotation_quality,
      'period': period*nm_per_px,
      'period confidence': period_confidence,
      'lumen width': white_width*nm_per_px
  }
  if return_extra:
    result['extra'] = {
          'autocorrelation': autocorrelation,
          'oriented_img': oriented_img
      }

  return result

def get_top_k(a, k):
  ind = np.argpartition(a, -k)[-k:]
  return a[ind]

def get_crops(img, distance_map, crop_size, N_sample):
  crop_r= np.sqrt(2)*crop_size / 2
  possible_positions_y, possible_positions_x  = np.where(distance_map >= crop_r)
  no_edge_mask = (possible_positions_y>crop_r) & \
   (possible_positions_x>crop_r) & \
   (possible_positions_y<(distance_map.shape[0]-crop_r)) & \
   (possible_positions_x<(distance_map.shape[1]-crop_r))

  possible_positions_x = possible_positions_x[no_edge_mask]
  possible_positions_y = possible_positions_y[no_edge_mask]
  N_available = len(possible_positions_x)
  positions_indices = np.random.choice(np.arange(N_available), min(N_sample, N_available), replace=False)

  for idx in positions_indices:
    yield img[possible_positions_y[idx]-crop_size//2:possible_positions_y[idx]+crop_size//2,possible_positions_x[idx]-crop_size//2:possible_positions_x[idx]+crop_size//2].copy()

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
    
def select_samples(granum_image, granum_mask, crop_size=96, n_samples=64, granum_fraction_min=1.0, variance_p=2):
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

def calculate_distance_map(mask):
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=False)
    distance_map_padded = ndimage.distance_transform_edt(padded)
    return distance_map_padded[1:-1,1:-1]


def measure_object(
  img, mask,
  nm_per_px=1, n_tries = 3,
  direction_thr_min = 0.07, direction_thr_enough = 0.1,
  crop_size = 200,
  **kwargs): 

    distance_map = calculate_distance_map(mask)
    crop_size = min(crop_size, int(min(get_top_k(distance_map.flatten(), n_tries)*0.5**0.5)))
  
    direction_confidence = 0
    best_stripes_data = {}
    for i, img_crop in enumerate(get_crops(img, distance_map, crop_size, n_tries)):
        stripes_data = process_img_crop(img_crop, nm_per_px=nm_per_px)
        if stripes_data['direction confidence'] >= direction_confidence:
            best_stripes_data = deepcopy(stripes_data)
            direction_confidence = stripes_data['direction confidence']
        if direction_confidence > direction_thr_enough:
            break

    result = best_stripes_data

    if direction_confidence >= direction_thr_min:
    
        mask_oriented = rotate(mask, 90-result['direction'], resize=True).astype('bool')
        idx_begin_x, idx_end_x = np.where(np.any(mask_oriented, axis=0))[0][np.array([0, -1])]
        idx_begin_y, idx_end_y = np.where(np.any(mask_oriented, axis=1))[0][np.array([0, -1])]
        result['mask_oriented'] = mask_oriented[idx_begin_y:idx_end_y, idx_begin_x:idx_end_x]
        result['img_oriented'] = rotate(img, 90-result['direction'], resize=True)[idx_begin_y:idx_end_y, idx_begin_x:idx_end_x]

  #   measurements = measure_granum_shape(result['mask_oriented'], nm_per_px=nm_per_px, oriented=True)
  # else:
  #   measurements = measure_granum_shape(mask, nm_per_px=nm_per_px, oriented=False)
  
  # result.update(**measurements)
  # N_layers = result['height'] / result['period']
  # if np.isfinite(N_layers):
  #   N_layers = round(N_layers)

    return result

# def measure_object(
#     img, mask,
#     nm_per_px=1, n_tries = 3,
#     direction_thr_min = 0.07, direction_thr_enough = 0.1,
#     crop_size = 200,
#     **kwargs): 

#     distance_map = calculate_distance_map(mask)
#     crop_size = min(crop_size, int((min(get_top_k(distance_map.flatten(), n_tries)*0.5)**0.5)))
  
#     direction_confidence = 0
#     best_stripes_data = {}
#     for i, img_crop in enumerate(select_samples(img, mask, crop_size=crop_size, n_samples=n_tries)):
#         stripes_data = process_img_crop(img_crop, nm_per_px=nm_per_px)
#         if stripes_data['direction_confidence'] >= direction_confidence:
#             best_stripes_data = deepcopy(stripes_data)
#             direction_confidence = stripes_data['direction_confidence']
#         if direction_confidence > direction_thr_enough:
#             break

#     result = best_stripes_data

#     if direction_confidence >= direction_thr_min:
    
#         mask_oriented = rotate(mask, 90-result['direction'], resize=True).astype('bool')
#         idx_begin_x, idx_end_x = np.where(np.any(mask_oriented, axis=0))[0][np.array([0, -1])]
#         idx_begin_y, idx_end_y = np.where(np.any(mask_oriented, axis=1))[0][np.array([0, -1])]
#         result['mask_oriented'] = mask_oriented[idx_begin_y:idx_end_y, idx_begin_x:idx_end_x]
#         result['img_oriented'] = rotate(img, 90-result['direction'], resize=True)[idx_begin_y:idx_end_y, idx_begin_x:idx_end_x]

#   #   measurements = measure_granum_shape(result['mask_oriented'], nm_per_px=nm_per_px, oriented=True)
#   # else:
#   #   measurements = measure_granum_shape(mask, nm_per_px=nm_per_px, oriented=False)
  
#   # result.update(**measurements)
#   # N_layers = result['height'] / result['period']
#   # if np.isfinite(N_layers):
#   #   N_layers = round(N_layers)

#     return result #{**measurements, **best_stripes_data, 'N layers': N_layers}