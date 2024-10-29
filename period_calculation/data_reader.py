import pandas as pd
import numpy as np
import skimage.io
from pathlib import Path
import torch
import scipy
from PIL import Image, ImageFilter, ImageChops
# from config import model_config
from period_calculation.config import model_config

# Function to add Gaussian noise

def add_microscope_noise(base_image_as_numpy, noise_intensity):
    ###### The code below is for adding noise to the image
    # noise intensity is a number between 0 and 1
    # --- priginal implementation was provided by Michał Bykowski
    # --- and adapted
    # This routine works with PIL images and numpy internally (changing formats as it goes)
    # but the input and output are numpy arrays

    def add_noise(image, mean=0, std_dev=50):  # std_dev impacts the amount of noise
        # Generating noise
        noise = np.random.normal(mean, std_dev, (image.height, image.width))
        # Adding noise to the image
        noisy_image = np.array(image) + noise
        # Ensuring the values remain within valid grayscale range
        noisy_image = np.clip(noisy_image, 0, 255)
        return Image.fromarray(noisy_image.astype('uint8'))


    base_image = Image.fromarray(base_image_as_numpy)
    gray_value = 128
    gray = Image.new('L', base_image.size, color=gray_value)


    gray = add_noise(gray, std_dev=noise_intensity * 76)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=3))
    gray = add_noise(gray, std_dev=noise_intensity * 23)
    gray = gray.filter(ImageFilter.GaussianBlur(radius=2))
    gray = add_noise(gray, std_dev=noise_intensity * 15)

    # soft light works as in Photoshop
    # Superimposes two images on top of each other using the Soft Light algorithm
    result = ImageChops.soft_light(base_image, gray)

    return np.array(result)

def detect_boundaries(mask, axis):
    # calculate the boundaries of the mask
    #axis = 0 results in x_from, x_to
    #axis = 1 results in y_from, y_to


    sum = mask.sum(axis=axis)

    ind_from = min(sum.nonzero()[0])
    ind_to = max(sum.nonzero()[0])
    return ind_from, ind_to

def add_symmetric_filling_beyond_mask(img, mask):
    for x in range(img.shape[1]):
        if sum(mask[:, x]) != 0:   #if there is at least one nonzero index
            nonzero_indices = mask[:, x].nonzero()[0]

            y_min = min(nonzero_indices)
            y_max = max(nonzero_indices)

            if y_max == y_min:   #there is only one point
                img[:, x] = img[y_min, x]
            else:
                next = y_min + 1
                step = +1  # we start by going upwards
                for y in reversed(range(y_min)):
                    img[y, x] = img[next, x]
                    if next == y_max or next == y_min:   #we hit the boundaries - we reverse
                        step *= -1   #reverse direction
                    next += step

                next = y_max - 1
                step = -1  # we start by going downwards
                for y in range(y_max + 1, img.shape[0]): #we hit the boundaries - we reverse
                    img[y, x] = img[next, x]
                    if next == y_max or next == y_min:
                        step *= -1  # reverse direction
                    next += step
    return img
class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self,
                 model = None,
                 transforms=[],
                 #### distortions during training ####
                 hv_symmetry=True,  # True or False

                 min_horizontal_subsampling = 50,  # None to turn off; or minimal percentage of horizontal size of the image
                 min_vertical_subsampling = 70,  # None to turn off; or minimal percentage of vertical size of the image
                 max_random_tilt = 3,  # None to turn off; or maximum tilt in degrees
                 max_add_colors_to_histogram = 10,  # 0 to turn off; or points of the histogram to be added
                 max_remove_colors_from_histogram = 30,  # 0 to turn off; or points of the histogram to be removed
                 max_noise_intensity = 3.0,  # 0.0 to turn off; or max intensity of the noise

                 gaussian_phase_transforms_epoch=None,  # None to turn off; or number of the epoch when the gaussian phase starts
                 min_horizontal_subsampling_gaussian_phase = 30,  # None to turn off; or minimal percentage of horizontal size of the image
                 min_vertical_subsampling_gaussian_phase = 70,  # None to turn off; or minimal percentage of vertical size of the image
                 max_random_tilt_gaussian_phase = 2,  # None to turn off; or maximum tilt in degrees
                 max_add_colors_to_histogram_gaussian_phase = 10,  # 0 to turn off; or points of the histogram to be added
                 max_remove_colors_from_histogram_gaussian_phase = 60,  # 0 to turn off; or points of the histogram to be removed
                 max_noise_intensity_gaussian_phase = 3.5,  # 0.0 to turn off; or max intensity of the noise

                 #### controling variables ####
                 transform_level=2,   # 0 - no transforms, 1 - only the basic transform, 2 - all transforms, -1 - subsampling for high images
                 retain_raw_images=False,
                 retain_masks=False):


        self.model = model  # we need that to check epoch number during training

        self.hv_symmetry = hv_symmetry

        self.min_horizontal_subsampling = min_horizontal_subsampling
        self.min_vertical_subsampling = min_vertical_subsampling
        self.max_random_tilt = max_random_tilt
        self.max_add_colors_to_histogram = max_add_colors_to_histogram
        self.max_remove_colors_from_histogram = max_remove_colors_from_histogram
        self.max_noise_intensity = max_noise_intensity

        self.gaussian_phase_transforms_epoch = gaussian_phase_transforms_epoch
        self.min_horizontal_subsampling_gaussian_phase = min_horizontal_subsampling_gaussian_phase
        self.min_vertical_subsampling_gaussian_phase = min_vertical_subsampling_gaussian_phase
        self.max_random_tilt_gaussian_phase = max_random_tilt_gaussian_phase
        self.max_add_colors_to_histogram_gaussian_phase = max_add_colors_to_histogram_gaussian_phase
        self.max_remove_colors_from_histogram_gaussian_phase = max_remove_colors_from_histogram_gaussian_phase
        self.max_noise_intensity_gaussian_phase = max_noise_intensity_gaussian_phase

        self.image_height = model_config['image_height']
        self.image_width = model_config['image_width']

        self.transform_level = transform_level
        self.retain_raw_images = retain_raw_images
        self.retain_masks = retain_masks
        self.transforms = transforms


    def get_image_and_mask(self, row):
        raise NotImplementedError("Subclass needs to implement this method")

    def load_and_transform_image_and_mask(self, row):
        img, mask = self.get_image_and_mask(row)

        angle = row['angle']
        #check if gaussian phase is on
        if self.gaussian_phase_transforms_epoch is not None and self.model.current_epoch >= self.gaussian_phase_transforms_epoch:
            max_random_tilt = self.max_random_tilt_gaussian_phase
            max_noise_intensity = self.max_noise_intensity_gaussian_phase
            min_horizontal_subsampling = self.min_horizontal_subsampling_gaussian_phase
            min_vertical_subsampling = self.min_vertical_subsampling_gaussian_phase
            max_add_colors_to_histogram = self.max_add_colors_to_histogram_gaussian_phase
            max_remove_colors_from_histogram = self.max_remove_colors_from_histogram_gaussian_phase
        else:
            max_random_tilt = self.max_random_tilt
            max_noise_intensity = self.max_noise_intensity
            min_horizontal_subsampling = self.min_horizontal_subsampling
            min_vertical_subsampling = self.min_vertical_subsampling
            max_add_colors_to_histogram = self.max_add_colors_to_histogram
            max_remove_colors_from_histogram = self.max_remove_colors_from_histogram






        if self.transform_level >= 2 and max_random_tilt is not None:
            ####### RANDOM TILT
            angle += np.random.uniform(-max_random_tilt, max_random_tilt)

        img = scipy.ndimage.rotate(img, 90 - angle, reshape=True, order=3)  # HORIZONTAL POSITION
        ###the part of the image that is added after rotation is all black (0s)
        mask = scipy.ndimage.rotate(mask, 90 - angle, reshape=True, order = 0)  # HORIZONTAL POSITION
                    #order = 0 is the nearest neighbor interpolation, so the mask is not interpolated

        ############# CROP
        x_from, x_to = detect_boundaries(mask, axis=0)
        y_from, y_to = detect_boundaries(mask, axis=1)

        #crop the image to the verical and horizontal limits.
        img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
        mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]


        img_raw = img.copy()


        if self.transform_level >= 2:
            ########## ADDING NOISE

            if max_noise_intensity > 0.0:
                noise_intensity = np.random.random() * max_noise_intensity
                noisy_img = add_microscope_noise(img, noise_intensity=noise_intensity)
                img[mask] = noisy_img[mask]

        if self.transform_level == -1:
            #special case where we take at most 300 middle pixels from the image
            # (vertical subsampling)
            # to handle very latge images correctly
            x_from, x_to = detect_boundaries(mask, axis=0)
            y_from, y_to = detect_boundaries(mask, axis=1)

            y_size = y_to - y_from + 1

            random_size = 300   #not so random, ay?

            if y_size > random_size:
                random_start = y_size // 2 - random_size // 2

                y_from = random_start
                y_to = random_start + random_size - 1

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

                # recrop the image if necessary
                # -- even after only horizontal subsampling it may be necessary to recrop the image

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

        if self.transform_level >= 1:
            ############## HORIZONTAL SUBSAMPLING
            if min_horizontal_subsampling is not None:
                x_size = x_to - x_from + 1

                # add some random horizontal shift
                random_size = np.random.randint(x_size * min_horizontal_subsampling / 100.0, x_size + 1)
                random_start = np.random.randint(0, x_size - random_size + 1) + x_from

                img = img[:, random_start:(random_start + random_size)]
                mask = mask[:, random_start:(random_start + random_size)]

            ############ VERTICAL SUBSAMPLING
            if min_vertical_subsampling is not None:

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                y_size = y_to - y_from + 1

                random_size = np.random.randint(y_size * min_vertical_subsampling / 100.0, y_size + 1)
                random_start = np.random.randint(0, y_size - random_size + 1) + y_from

                y_from = random_start
                y_to = random_start + random_size - 1

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

            if min_horizontal_subsampling is not None or min_vertical_subsampling is not None:
                #recrop the image if necessary
                # -- even after only horizontal subsampling it may be necessary to recrop the image

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]


        ######### ADD SYMMETRIC FILLING OF THE IMAGE BEYOND THE MASK
        #img = add_symmetric_filling_beyond_mask(img, mask)
        #This leaves holes in the image, so we will not use it

        #plt.imshow(img)
        #plt.show()
        ######### HORIZONTAL AND VERTICAL SYMMETRY.
        # When superimposed, the result is 180 degree rotation
        if self.transform_level >= 1 and self.hv_symmetry:
            for axis in range(2):
                if np.random.randint(0, 2) % 2 == 0:
                    img = np.flip(img, axis = axis)
                    mask = np.flip(mask, axis = axis)
            #plt.imshow(img)
            #plt.show()

        if self.transform_level >= 2 and (max_add_colors_to_histogram > 0 or max_remove_colors_from_histogram > 0):
            lower_bound = np.random.randint(-max_add_colors_to_histogram, max_remove_colors_from_histogram + 1)
            upper_bound = np.random.randint(255 - max_remove_colors_from_histogram, 255 + max_add_colors_to_histogram + 1)
            # first clip the values outstanding from the range (lower_bound -- upper_bound)
            img[mask] = np.clip(img[mask], lower_bound, upper_bound)
            # the range (lower_bound -- upper_bound) gets mapped to the range (0--255)
            # but only in a portion of the image where mask = True
            img[mask] = np.interp(img[mask], (lower_bound, upper_bound), (0, 255)).astype(np.uint8)

        #### since preserve_range in skimage.transform.resize is set to False, the image
        #### will be converted to float. Consult:
        # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
        # https://scikit-image.org/docs/dev/user_guide/data_types.html

        # In our case the image gets conveted to floats ranging 0-1
        old_height = img.shape[0]
        img = skimage.transform.resize(img, (self.image_height, self.image_width), order=3)
        new_height = img.shape[0]
        mask = skimage.transform.resize(mask, (self.image_height, self.image_width), order=0, preserve_range=True)
           # order = 0 is the nearest neighbor interpolation, so the mask is not interpolated

        scale_factor = new_height / old_height


        #plt.imshow(img)
        #plt.show()
        #plt.imshow(mask)
        #plt.show()
        return img, mask, scale_factor, img_raw

    def get_annotations_row(self, idx):
        raise NotImplementedError("Subclass needs to implement this method")

    def __getitem__(self, idx):
        row = self.get_annotations_row(idx)

        image, mask, scale_factor, image_raw = self.load_and_transform_image_and_mask(row)

        image_data = {
            'image': image,
        }

        for transform in self.transforms:
            image_data = transform(**image_data)
            # transform operates on image field ONLY of image_data, and returns a dictionary with the same keys

        ret_dict = {
            'image': image_data['image'],
            'period_px': torch.tensor(row['period_nm'] * scale_factor * row['px_per_nm'], dtype=torch.float32),
            'filename': row['granum_image'],
            'px_per_nm': row['px_per_nm'],
            'scale': scale_factor,         # the scale factor is used to calculate the true period error
                                          # (before scale) in losses and metrics
            'neutral': -self.transforms[0].mean/self.transforms[0].std   #value of 0 after the scale transform
        }

        if self.retain_raw_images:
            ret_dict['image_raw'] = image_raw

        if self.retain_masks:
            ret_dict['mask'] = mask

        return ret_dict

    def __len__(self):
        raise NotImplementedError("Subclass needs to implement this method")

class ImageDataset(AbstractDataset):
    def __init__(self, annotations, data_dir: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = Path(data_dir)

        self.id = 1

        if isinstance(annotations, str):
            annotations = data_dir / annotations  #make it a Path object relative to data_dir

        if isinstance(annotations, Path):
            self.annotations = pd.read_csv(data_dir / annotations)
            no_period = ['27_k7 [1]_4.png']
            del_img = ['38_k42[1]_19.png', 'n6363_araLL_60kx_6 [1]_0.png', '27_hs8 [1]_5.png', '27_k7 [1]_20.png',
                       'F1_1_60kx_01 [1]_2.png']
            self.annotations = self.annotations[~self.annotations['granum_image'].isin(no_period)]
            self.annotations = self.annotations[~self.annotations['granum_image'].isin(del_img)]
        else:
            self.annotations = annotations

    def get_image_and_mask(self, row):
        filename = row['granum_image']
        img_path = self.data_dir / filename
        img_raw = skimage.io.imread(img_path)

        img = img_raw[:, :, 0]   # all three channels are equal, with the exception
        # of the last channel which is the full blue (0,0,255) for outside the mask (so blue channel is 255, red and green are 0)
        mask = (img_raw != (0, 0, 255)).any(axis=2)
        return img, mask

    def get_annotations_row(self, idx):
        row = self.annotations.iloc[idx].to_dict()
        row['idx'] = idx
        return row

    def __len__(self):
        return len(self.annotations)

class ArtificialDataset(AbstractDataset):
    def __init__(self,
                 min_period = 20,
                 max_period = 140,
                 white_fraction_min = 0.15,
                 white_fraction_max=0.45,

                 noise_min_sd = 0.0,
                 noise_max_sd = 100.0,
                 noise_max_sd_everywhere = 20.0,  # 20.0
                 leftovers_max = 5,

                 get_real_masks_dataset = None,   #None or instance of ImageDataset
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = 0
        self.min_period = min_period
        self.max_period = max_period
        self.white_fraction_min = white_fraction_min
        self.white_fraction_max = white_fraction_max

        self.receptive_field_height = model_config['receptive_field_height']
        self.stride_height = model_config['stride_height']
        self.receptive_field_width = model_config['receptive_field_width']
        self.stride_width = model_config['stride_width']

        self.noise_min_sd = noise_min_sd
        self.noise_max_sd = noise_max_sd
        self.noise_max_sd_everywhere = noise_max_sd_everywhere

        self.leftovers_max = leftovers_max

        self.get_real_masks_dataset = get_real_masks_dataset


    def get_image_and_mask(self, row):
        # generate a rectangular image of black and white horizontal stripes
        # with black stripes varying with white stripes

        period_px = row['period_nm'] * row['px_per_nm']
        # white occupying 5-20 % of a total period (white+black)
        white_px = np.random.randint(period_px * self.white_fraction_min, period_px * self.white_fraction_max + 1)


        # mask is rectangle of True values
        img = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        mask = np.ones((self.image_height, self.image_width), dtype=bool)
        black_px = period_px - white_px
        random_start = np.random.randint(0, period_px+1)
        for i in range(self.image_height):
            if (random_start+i) % (black_px + white_px) < black_px:
                # sample width with random numbers from 0 to 101
                img[i, :] = np.random.randint(0, 101, self.image_width)
            else:
                #sample width with random numbers from 156 to 255
                img[i, :] = np.random.randint(156, 256, self.image_width)

        if self.noise_max_sd_everywhere > self.noise_min_sd:
            sd = np.random.uniform(self.noise_min_sd, self.noise_max_sd_everywhere)
            noise = np.random.normal(0, sd, (self.image_height, self.image_width))
            img = np.clip(img+noise.astype(img.dtype), 0, 255)

        if self.noise_max_sd > self.noise_min_sd:
            # there is also a metagrid in the image
            # consisting of overlapping receptive fields of size 190x42
            # with stride 64x4
            # the metagrid is 5x102
            overlapping_fields_count_height = (self.image_height - self.receptive_field_height) // self.stride_height + 1
            overlapping_fields_count_width = (self.image_width - self.receptive_field_width) // self.stride_width + 1


            sd = np.random.uniform(self.noise_min_sd, self.noise_max_sd)
            noise = np.random.normal(0, sd, (self.image_height, self.image_width))

            #there will be some left-over metagrid rectangles
            leftovers_count = np.random.randint(1, self.leftovers_max)
            for i in range(leftovers_count):
                metagrid_row = np.random.randint(0, overlapping_fields_count_height)
                metagrid_col = np.random.randint(0, overlapping_fields_count_width)
                #zero-out the noise inside the selected metagrid
                noise[metagrid_row * self.stride_height:metagrid_row * self.stride_height + self.receptive_field_height + 1, \
                      metagrid_col * self.stride_width :metagrid_col * self.stride_width + self.receptive_field_width + 1] = 0

            #add noise to the image
            img = np.clip(img+noise.astype(img.dtype), 0, 255)

            if self.get_real_masks_dataset is not None:
                ret_dict = self.get_real_masks_dataset.__getitem__(row['idx'] % len(self.get_real_masks_dataset))
                mask = ret_dict['mask']   #this mask is already sized target height-by-width

                img[mask == False] = 0

        return img, mask

    def get_annotations_row(self, idx):
        return {'idx': idx,
                'period_nm': np.random.randint(self.min_period, self.max_period),
                'px_per_nm': 1.0,
                'granum_image': 'artificial_%d.png' % idx,
                'angle': 90}


    def __len__(self):
        return 237 # number of samples as in real data in the train set (70% of 339 is 237,3)

    
class AdHocDataset(AbstractDataset):
    def __init__(self, images_masks_pxpernm: list[tuple[np.ndarray, np.ndarray, float]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = images_masks_pxpernm
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask, px_per_nm = self.data[idx]

        image, mask, scale_factor, image_raw = self.load_and_transform_image_and_mask(image, mask)

        image_data = {
            'image': image,
        }

        for transform in self.transforms:
            image_data = transform(**image_data)
            # transform operates on image field ONLY of image_data, and returns a dictionary with the same keys

        ret_dict = {
            'image': image_data['image'],
            'period_px': torch.tensor(0, dtype=torch.float32),
            'filename': str(idx),
            'px_per_nm': px_per_nm,
            'scale': scale_factor,         # the scale factor is used to calculate the true period error
                                          # (before scale) in losses and metrics
            'neutral': -self.transforms[0].mean/self.transforms[0].std   #value of 0 after the scale transform
        }

        if self.retain_raw_images:
            ret_dict['image_raw'] = image_raw

        if self.retain_masks:
            ret_dict['mask'] = mask

        return ret_dict
        
        
    def load_and_transform_image_and_mask(self, img, mask):
        
        angle = 90
        #check if gaussian phase is on
        if self.gaussian_phase_transforms_epoch is not None and self.model.current_epoch >= self.gaussian_phase_transforms_epoch:
            max_random_tilt = self.max_random_tilt_gaussian_phase
            max_noise_intensity = self.max_noise_intensity_gaussian_phase
            min_horizontal_subsampling = self.min_horizontal_subsampling_gaussian_phase
            min_vertical_subsampling = self.min_vertical_subsampling_gaussian_phase
            max_add_colors_to_histogram = self.max_add_colors_to_histogram_gaussian_phase
            max_remove_colors_from_histogram = self.max_remove_colors_from_histogram_gaussian_phase
        else:
            max_random_tilt = self.max_random_tilt
            max_noise_intensity = self.max_noise_intensity
            min_horizontal_subsampling = self.min_horizontal_subsampling
            min_vertical_subsampling = self.min_vertical_subsampling
            max_add_colors_to_histogram = self.max_add_colors_to_histogram
            max_remove_colors_from_histogram = self.max_remove_colors_from_histogram


        if self.transform_level >= 2 and max_random_tilt is not None:
            ####### RANDOM TILT
            angle += np.random.uniform(-max_random_tilt, max_random_tilt)
        

        img = scipy.ndimage.rotate(img, 90 - angle, reshape=True, order=3)  # HORIZONTAL POSITION
        ###the part of the image that is added after rotation is all black (0s)
        mask = scipy.ndimage.rotate(mask, 90 - angle, reshape=True, order = 0)  # HORIZONTAL POSITION
                    #order = 0 is the nearest neighbor interpolation, so the mask is not interpolated

        ############# CROP
        x_from, x_to = detect_boundaries(mask, axis=0)
        y_from, y_to = detect_boundaries(mask, axis=1)

        #crop the image to the verical and horizontal limits.
        img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
        mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]


        img_raw = img.copy()


        if self.transform_level >= 2:
            ########## ADDING NOISE

            if max_noise_intensity > 0.0:
                noise_intensity = np.random.random() * max_noise_intensity
                noisy_img = add_microscope_noise(img, noise_intensity=noise_intensity)
                img[mask] = noisy_img[mask]

        if self.transform_level == -1:
            #special case where we take at most 300 middle pixels from the image
            # (vertical subsampling)
            # to handle very latge images correctly
            x_from, x_to = detect_boundaries(mask, axis=0)
            y_from, y_to = detect_boundaries(mask, axis=1)

            y_size = y_to - y_from + 1

            random_size = 300   #not so random, ay?

            if y_size > random_size:
                random_start = y_size // 2 - random_size // 2

                y_from = random_start
                y_to = random_start + random_size - 1

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

                # recrop the image if necessary
                # -- even after only horizontal subsampling it may be necessary to recrop the image

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

        if self.transform_level >= 1:
            ############## HORIZONTAL SUBSAMPLING
            if min_horizontal_subsampling is not None:
                x_size = x_to - x_from + 1

                # add some random horizontal shift
                random_size = np.random.randint(x_size * min_horizontal_subsampling / 100.0, x_size + 1)
                random_start = np.random.randint(0, x_size - random_size + 1) + x_from

                img = img[:, random_start:(random_start + random_size)]
                mask = mask[:, random_start:(random_start + random_size)]

            ############ VERTICAL SUBSAMPLING
            if min_vertical_subsampling is not None:

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                y_size = y_to - y_from + 1

                random_size = np.random.randint(y_size * min_vertical_subsampling / 100.0, y_size + 1)
                random_start = np.random.randint(0, y_size - random_size + 1) + y_from

                y_from = random_start
                y_to = random_start + random_size - 1

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

            if min_horizontal_subsampling is not None or min_vertical_subsampling is not None:
                #recrop the image if necessary
                # -- even after only horizontal subsampling it may be necessary to recrop the image

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]


        ######### ADD SYMMETRIC FILLING OF THE IMAGE BEYOND THE MASK
        #img = add_symmetric_filling_beyond_mask(img, mask)
        #This leaves holes in the image, so we will not use it

        #plt.imshow(img)
        #plt.show()
        ######### HORIZONTAL AND VERTICAL SYMMETRY.
        # When superimposed, the result is 180 degree rotation
        if self.transform_level >= 1 and self.hv_symmetry:
            for axis in range(2):
                if np.random.randint(0, 2) % 2 == 0:
                    img = np.flip(img, axis = axis)
                    mask = np.flip(mask, axis = axis)
            #plt.imshow(img)
            #plt.show()

        if self.transform_level >= 2 and (max_add_colors_to_histogram > 0 or max_remove_colors_from_histogram > 0):
            lower_bound = np.random.randint(-max_add_colors_to_histogram, max_remove_colors_from_histogram + 1)
            upper_bound = np.random.randint(255 - max_remove_colors_from_histogram, 255 + max_add_colors_to_histogram + 1)
            # first clip the values outstanding from the range (lower_bound -- upper_bound)
            img[mask] = np.clip(img[mask], lower_bound, upper_bound)
            # the range (lower_bound -- upper_bound) gets mapped to the range (0--255)
            # but only in a portion of the image where mask = True
            img[mask] = np.interp(img[mask], (lower_bound, upper_bound), (0, 255)).astype(np.uint8)

        #### since preserve_range in skimage.transform.resize is set to False, the image
        #### will be converted to float. Consult:
        # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
        # https://scikit-image.org/docs/dev/user_guide/data_types.html

        # In our case the image gets conveted to floats ranging 0-1
        old_height = img.shape[0]
        img = skimage.transform.resize(img, (self.image_height, self.image_width), order=3)
        new_height = img.shape[0]
        mask = skimage.transform.resize(mask, (self.image_height, self.image_width), order=0, preserve_range=True)
           # order = 0 is the nearest neighbor interpolation, so the mask is not interpolated

        scale_factor = new_height / old_height


        #plt.imshow(img)
        #plt.show()
        #plt.imshow(mask)
        #plt.show()
        return img, mask, scale_factor, img_raw

    
class AdHocDataset2(AbstractDataset):
    def __init__(self, images_masks_pxpernm: list[tuple[np.ndarray, np.ndarray, float]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = images_masks_pxpernm
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask, px_per_nm = self.data[idx]

        image, mask, scale_factor, image_raw = self.load_and_transform_image_and_mask(image, mask)

        image_data = {
            'image': image,
        }

        for transform in self.transforms:
            image_data = transform(**image_data)
            # transform operates on image field ONLY of image_data, and returns a dictionary with the same keys

        ret_dict = {
            'image': image_data['image'],
            'scale': scale_factor,         # the scale factor is used to calculate the true period error
                                          # (before scale) in losses and metrics
            'neutral': -self.transforms[0].mean/self.transforms[0].std   #value of 0 after the scale transform
        }

        return ret_dict
        
        
    def load_and_transform_image_and_mask(self, img, mask):
 
        img_raw = img.copy()

        if self.transform_level == -1:
            #special case where we take at most 300 middle pixels from the image
            # (vertical subsampling)
            # to handle very latge images correctly
            x_from, x_to = detect_boundaries(mask, axis=0)
            y_from, y_to = detect_boundaries(mask, axis=1)

            y_size = y_to - y_from + 1

            max_size = 300

            if y_size > max_size:
                random_start = y_size // 2 - max_size // 2

                y_from = random_start
                y_to = random_start + max_size - 1

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

                # recrop the image if necessary
                # -- even after only horizontal subsampling it may be necessary to recrop the image

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]


        #### since preserve_range in skimage.transform.resize is set to False, the image
        #### will be converted to float. Consult:
        # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
        # https://scikit-image.org/docs/dev/user_guide/data_types.html

        # In our case the image gets conveted to floats ranging 0-1
        old_height = img.shape[0]
        img = skimage.transform.resize(img, (self.image_height, self.image_width), order=3)
        new_height = img.shape[0]
        mask = skimage.transform.resize(mask, (self.image_height, self.image_width), order=0, preserve_range=True)
           # order = 0 is the nearest neighbor interpolation, so the mask is not interpolated

        scale_factor = new_height / old_height

        return img, mask, scale_factor, img_raw

class AdHocDataset3(AbstractDataset):
    def __init__(self, images_and_masks: list[tuple[np.ndarray, np.ndarray]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = images_and_masks
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, mask = self.data[idx]

        image, mask, scale_factor = self.load_and_transform_image_and_mask(image, mask)

        image_data = {
            'image': image,
        }

        for transform in self.transforms:
            image_data = transform(**image_data)
            # transform operates on image field ONLY of image_data, and returns a dictionary with the same keys

        ret_dict = {
            'image': image_data['image'],
            'scale': scale_factor,         # the scale factor is used to calculate the true period error
                                          # (before scale) in losses and metrics
               #value of 0 after the scale transform
        }

        return ret_dict
        
        
    def load_and_transform_image_and_mask(self, img, mask):
 
        if self.transform_level == -1:
            #special case where we take at most 300 middle pixels from the image
            # (vertical subsampling)
            # to handle very latge images correctly
            x_from, x_to = detect_boundaries(mask, axis=0)
            y_from, y_to = detect_boundaries(mask, axis=1)

            y_size = y_to - y_from + 1

            max_size = 300

            if y_size > max_size:
                random_start = y_size // 2 - max_size // 2

                y_from = random_start
                y_to = random_start + max_size - 1

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]

                # recrop the image if necessary
                # -- even after only horizontal subsampling it may be necessary to recrop the image

                x_from, x_to = detect_boundaries(mask, axis=0)
                y_from, y_to = detect_boundaries(mask, axis=1)

                img = img[y_from:(y_to + 1), x_from:(x_to + 1)]
                mask = mask[y_from:(y_to + 1), x_from:(x_to + 1)]


        #### since preserve_range in skimage.transform.resize is set to False, the image
        #### will be converted to float. Consult:
        # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
        # https://scikit-image.org/docs/dev/user_guide/data_types.html

        # In our case the image gets conveted to floats ranging 0-1
        old_height = img.shape[0]
        img = skimage.transform.resize(img, (self.image_height, self.image_width), order=3)
        new_height = img.shape[0]
        mask = skimage.transform.resize(mask, (self.image_height, self.image_width), order=0, preserve_range=True)
           # order = 0 is the nearest neighbor interpolation, so the mask is not interpolated

        scale_factor = new_height / old_height

        return img, mask, scale_factor