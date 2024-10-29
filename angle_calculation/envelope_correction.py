import numpy as np
import scipy

def detect_boundaries(mask, axis):
    # calculate the boundaries of the mask
    #axis = 0 results in x_from, x_to
    #axis = 1 results in y_from, y_to


    sum = mask.sum(axis=axis)

    ind_from = min(sum.nonzero()[0])
    ind_to = max(sum.nonzero()[0])
    return ind_from, ind_to

def area(mask):
    x1, y1 = detect_boundaries(mask, 0)
    a = y1 - x1
    x2, y2 = detect_boundaries(mask, 1)
    b = y2 - x2

    return (a * b, x1, y1, x2, y2)

def calculate_best_angle_from_mask(mask, angles=np.arange(-10,10,0.5)):
        areas = []
        for angle in angles:  
            rotated_mask = scipy.ndimage.rotate(mask, angle, reshape=True, order = 0)  # order = 0 is the nearest neighbor interpolation, so the mask is not interpolated
            this_area = area(rotated_mask)
            areas.append(this_area[0])
            
        best_angle = angles[np.argmin(areas)]
        return best_angle

        