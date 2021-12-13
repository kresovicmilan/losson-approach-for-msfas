#import spectral as sp
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time
from pathlib import Path
import os
import random
import pickle
from PIL import Image
import cv2
#from skimage.feature.texture import greycomatrix_channels_extened
import multiprocessing as mp
from functools import partial
from ccm import *

# Leave it as only .raw files, because of code logic
FILE_EXTENSIONS = [
    '.raw'
]

IMG_FILE_EXTENSIONS = {
    '.png'
}

######################## FILTER RULES ########################
# Each filter pixel position in CFA image for 3 filters
FILTER_CONDITIONS_3 = [
    [[lambda x: x % 2 != 0, lambda x: x % 2 == 0]], # blue: odd row, even col rule
    [[lambda x: x % 2 == 0, lambda x: x % 2 == 0], [lambda x: x % 2 != 0, lambda x: x % 2 != 0]], # green: row 1, col1 rule, row2, col2 rule
    [[lambda x: x % 2 == 0, lambda x: x % 2 != 0]] # red: row, col rule
]

FILTER_CONDITIONS_4 = [
    [[lambda x: x % 2 == 0, lambda x: x % 2 == 0]], # filter_1: even row, even col rule
    [[lambda x: x % 2 != 0, lambda x: x % 2 != 0]], # filter_2: odd row, odd col rule
    [[lambda x: x % 2 == 0, lambda x: x % 2 != 0]], # filter_3: even row, odd col rule
    [[lambda x: x % 2 != 0, lambda x: x % 2 == 0]]  # filter_4: odd row, even col rule
]

FILTER_CONDITIONS_5 = [
    [[lambda x: x % 4 == 0, lambda x: x % 4 == 0], [lambda x: x % 4 == 2, lambda x: x % 4 == 2]], #filter_1: %4==0 row and %4==0 col, %4==2 row and %4==2 col
    [[lambda x: x % 2 != 0, lambda x: x % 2 != 0]], # filter_2: odd row, odd col rule
    [[lambda x: x % 2 == 0, lambda x: x % 2 != 0]], # filter_3: even row, odd col rule
    [[lambda x: x % 2 != 0, lambda x: x % 2 == 0]],  # filter_4: odd row, even col rule
    [[lambda x: x % 4 == 2, lambda x: x % 4 == 0], [lambda x: x % 4 == 0, lambda x: x % 4 == 2]] #filter_5: %4==2 row and %4==0 col, %4==0 row and %4==2 col
]

FILTER_CONDITIONS_8 = []
##############################################################

GENERAL_DATASET_DIR = './datasets'                        # General directory with all datasets
HYTEXILA_DIR = './hytexila'                               # Directory of HyTexiLa dataset, without / at the end

#### CHANGE AT THE SAME TIME ####
NEW_DATASET_DIR = GENERAL_DATASET_DIR + '/5_filters'       # Directory of new dataset, without / at the end
NB_FILTERS = 5                                             # Number of filters
SUBIMG_SAMPLE = 8                                          # Sampling over width and height for each image (16 or 64 or ... subimages)
N_SUB = int(SUBIMG_SAMPLE*SUBIMG_SAMPLE)                   # Number of subsample images
SUBSAMPLE_DICT = "/subsample_dict_{}.pkl".format(N_SUB)   # Pickle file with subsample dictionary
SHOULD_SUB_OPTIMIZE = True                                 # If needed to take less samples, to speed up the process
if SHOULD_SUB_OPTIMIZE:
    N_SUB = 16
    SUBSAMPLE_DICT = "/optimized_" + SUBSAMPLE_DICT.split("/")[1]

if NB_FILTERS == 3:                                        # Filter conditions for specific filters case
    FILTER_CONDITIONS = FILTER_CONDITIONS_3
elif NB_FILTERS == 4:
    FILTER_CONDITIONS = FILTER_CONDITIONS_4
elif NB_FILTERS == 5:
    FILTER_CONDITIONS = FILTER_CONDITIONS_5
elif NB_FILTERS == 8:
    FILTER_CONDITIONS = FILTER_CONDITIONS_8
else:
    print("FILTER CONDITIONS ARE NOT DEFINED")
    quit()
#################################

#### NO NEED FOR CHANGE ####
PEAK_FILTER_HEIGHT = 1                                     # Maximum peak of filter transmitance (Gaussian peak)
SIGMA_SCALAR = 4                                           # Distance between filter peaks
MAX_PIXEL_VALUE = 255                                      # Maximum pixel value, as some pixels are above 255
IMG_WIDTH = 1024                                           # Image width
IMG_HEIGHT = 1024                                          # Image height
SUBIMG_WIDTH = IMG_WIDTH//SUBIMG_SAMPLE                    # Width of subsampled image
SUBIMG_HEIGHT = IMG_HEIGHT//SUBIMG_SAMPLE                  # Height of subsampled image
CCM_DISTANCES = [1, 5]                                     # Distances for calculating CCM
#QUANTIZATION = [16, 32, 64, 128, 256]                      # Quantization levels
QUANTIZATION = [16]
############################

def is_image_file(filename, file_extensions=FILE_EXTENSIONS):
    return any(filename.endswith(extension) for extension in file_extensions)

def get_img_paths(hytexila_dir, new_dataset_dir):
    img_paths = []
    for root, _, fnames in sorted(os.walk(hytexila_dir)):
        for fname in fnames:
            if is_image_file(fname):
                # creating subdirectories in the new dataset directory
                Path(new_dataset_dir + root.replace(hytexila_dir, "")).mkdir(parents=True, exist_ok=True)
                path = os.path.join(root, fname.split(".raw")[0])
                img_paths.append(path)
    return img_paths

def gauss_filter(wavelength_bands, std=1, height=1, mean=0):
    from math import exp, pow
    variance = pow(std, 2)
    return height * exp(-pow(wavelength_bands-mean, 2)/(2*variance))

def dot_product_of_filter(img, filter, illuminant, max_pixel_value = 255):
    graychannel = np.empty([img.nrows, img.ncols], dtype=int)
    filtered_radiance = filter*illuminant
    summed_filtered_radiance = np.dot(filter, illuminant)
    for row in range(img.nrows):
        for col in range(img.ncols):
            # important to add round, because we don't want to just truncate the residual part of the float by using int cast
            graychannel[row, col] = int(round((np.dot(img[row, col], filtered_radiance)/summed_filtered_radiance)*max_pixel_value))
            if graychannel[row, col] > max_pixel_value:
                graychannel[row, col] = max_pixel_value
    return graychannel

def open_image(image_path):
    img = envi.open(image_path +'.hdr', image_path +'.raw')
    nb_bands = img.nbands
    start_lambda = img.bands.centers[0]
    end_lambda = img.bands.centers[-1]
    illuminant = list(map(int, img.metadata['illuminant'][0].split(" ; ")))
    return img, illuminant, start_lambda, end_lambda, nb_bands

def creating_filters(start_lambda, end_lambda, nb_bands, nb_filters, sigma_scalar = 4, peak_filter_height = 1):
    vector_gauss_filter = np.vectorize(gauss_filter)
    wavelength_bands = np.linspace(start_lambda, end_lambda, nb_bands)
    filters_offset = (end_lambda - start_lambda)/(nb_filters+1)
    filters_mean = [start_lambda + (i+1)*filters_offset for i in range(nb_filters)]
    # 2 sigma difference is not between peaks but between part where filters intersect
    std = filters_offset/sigma_scalar
    filters = [vector_gauss_filter(wavelength_bands, std=std, height=peak_filter_height, mean=fil_mean) for fil_mean in filters_mean]
    return filters

def display_filters(filters, start_lambda, end_lambda, nb_bands):
    wavelength_bands = np.linspace(start_lambda, end_lambda, nb_bands)
    for filter in filters:
        plt.plot(wavelength_bands, filter)
    plt.show()

def display_channel(channel):
    plt.imshow(channel, cmap='gray', vmin=0, vmax=255)
    plt.show()

def get_channels(img, filters, illuminant, max_pixel_value=255):
    channels = [dot_product_of_filter(img, filter, illuminant, max_pixel_value) for filter in filters]
    #rgb = np.dstack(list(reversed(channels)))
    return channels

def create_channels_dataset(img_paths):
    for image_path in img_paths:
        img, illuminant, start_lambda, end_lambda, nb_bands = open_image(image_path)
        filters = creating_filters(start_lambda, end_lambda, nb_bands, NB_FILTERS, SIGMA_SCALAR, PEAK_FILTER_HEIGHT)
        channels = get_channels(img, filters, illuminant, MAX_PIXEL_VALUE)
        save_grayscale_channels(image_path, channels)

def save_grayscale_channels(image_path, channels):
    new_image_path = NEW_DATASET_DIR + image_path.replace(HYTEXILA_DIR, "")
    for idx, channel in enumerate(channels):
        plt.imsave(new_image_path + "_filter_{}".format(idx) + ".png", channel, cmap='gray', vmin=0, vmax=255)

def one_image_example():
    image_path = 'hytexila/food/food_rice/food_rice'
    img, illuminant, start_lambda, end_lambda, nb_bands = open_image(image_path)
    filters = creating_filters(start_lambda, end_lambda, nb_bands, NB_FILTERS, SIGMA_SCALAR, PEAK_FILTER_HEIGHT)
    display_filters(filters, start_lambda, end_lambda, nb_bands)
    channels = get_channels(img, filters, illuminant, MAX_PIXEL_VALUE)

def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def subsample_images(file_name):
    subsample_dict = dict()
    subimages = [(x*SUBIMG_WIDTH, y*SUBIMG_HEIGHT, SUBIMG_WIDTH, SUBIMG_HEIGHT) for x in range(SUBIMG_SAMPLE) for y in range(SUBIMG_SAMPLE)]
    for root, _, _ in sorted(os.walk(HYTEXILA_DIR)):
        if root.count("/") == 3:
            sample_folder = root.replace(HYTEXILA_DIR, "")
            shuffled_list = subimages.copy()
            random.shuffle(shuffled_list)
            subsample_dict[sample_folder] = shuffled_list[:N_SUB]
    save_obj(subsample_dict, GENERAL_DATASET_DIR + file_name)

def get_cfa_image(root, fnames, filter_conditions):
    filters = get_filters_channels(root, fnames)
    cfa_image = np.zeros(filters[0].shape)
    for filter, condition in zip(filters, filter_conditions):
        for cond in condition:
            row_range = [x for x in list(range(0, IMG_HEIGHT)) if cond[0](x)]
            col_range = [x for x in list(range(0, IMG_WIDTH)) if cond[1](x)]
            for row in row_range:
                for col in col_range:
                    cfa_image[row, col] = filter[row, col]
    return cfa_image.astype(np.uint8)

def get_filters_channels(root, fnames):
    filters = []
    for fname in fnames:
        if is_image_file(fname, IMG_FILE_EXTENSIONS):
            filter = cv2.imread(root + "/" + fname, cv2.IMREAD_GRAYSCALE)
            #filter = plt.imread()
            filters.append(filter)
    return filters
    
def generate_cfa_images(NEW_DATASET_DIR, get_cfa_image, filter_conditions):
    for root, _, fnames in sorted(os.walk(NEW_DATASET_DIR)):
        if root.count("/") == 4:
            cfa_image = get_cfa_image(root, fnames, filter_conditions)
            plt.imsave(root + "/" + fnames[-1].split("filter")[0] + "cfa.png", cfa_image, cmap='gray', vmin=0, vmax=255)

def get_5_filter_cfa_ccms(image, n_distance, levels = 256):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angles_11 = angles_12 = angles_34 = angles_52 = angles_55 = [45, 135, 225, 315]
    angles_13 = angles_24 = angles_53 = [0, 180]
    angles_14 = angles_23 = angles_54 = [90, 270]
    angles_15 = [0, 90, 180, 270]

    ccm_all_11 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], angles_11, levels=levels, conditions = FILTER_CONDITIONS_5[0])
    ccm_all_22 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_5[1])
    ccm_all_33 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_5[2])
    ccm_all_44 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_5[3])
    ccm_all_55 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], angles_55, levels=levels, conditions = FILTER_CONDITIONS_5[4])

    ccm_all_12 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_12, levels=levels, conditions = FILTER_CONDITIONS_5[0])
    ccm_all_13 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_13, levels=levels, conditions = FILTER_CONDITIONS_5[0])
    ccm_all_14 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_14, levels=levels, conditions = FILTER_CONDITIONS_5[0])
    ccm_all_15 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], angles_15, levels=levels, conditions = FILTER_CONDITIONS_5[0])
    ccm_all_23 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_23, levels=levels, conditions = FILTER_CONDITIONS_5[1])
    ccm_all_24 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_24, levels=levels, conditions = FILTER_CONDITIONS_5[1])
    ccm_all_52 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_52, levels=levels, conditions = FILTER_CONDITIONS_5[4])
    ccm_all_34 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_34, levels=levels, conditions = FILTER_CONDITIONS_5[2])
    ccm_all_53 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_53, levels=levels, conditions = FILTER_CONDITIONS_5[4])
    ccm_all_54 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_54, levels=levels, conditions = FILTER_CONDITIONS_5[4])

    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    ccm_11 = normalize_over_distances(np.sum(ccm_all_11, axis=3))
    ccm_22 = normalize_over_distances(np.sum(ccm_all_22, axis=3))
    ccm_33 = normalize_over_distances(np.sum(ccm_all_33, axis=3))
    ccm_44 = normalize_over_distances(np.sum(ccm_all_44, axis=3))
    ccm_55 = normalize_over_distances(np.sum(ccm_all_55, axis=3))
    ccm_12 = normalize_over_distances(np.sum(ccm_all_12, axis=3))
    ccm_13 = normalize_over_distances(np.sum(ccm_all_13, axis=3))
    ccm_14 = normalize_over_distances(np.sum(ccm_all_14, axis=3))
    ccm_15 = normalize_over_distances(np.sum(ccm_all_15, axis=3))
    ccm_23 = normalize_over_distances(np.sum(ccm_all_23, axis=3))
    ccm_24 = normalize_over_distances(np.sum(ccm_all_24, axis=3))
    ccm_52 = normalize_over_distances(np.sum(ccm_all_52, axis=3))
    ccm_34 = normalize_over_distances(np.sum(ccm_all_34, axis=3))
    ccm_53 = normalize_over_distances(np.sum(ccm_all_53, axis=3))
    ccm_54 = normalize_over_distances(np.sum(ccm_all_54, axis=3))
    return [ccm_11, ccm_22, ccm_33, ccm_44, ccm_55, ccm_12, ccm_13, ccm_14, ccm_15, ccm_23, ccm_24, ccm_52, ccm_34, ccm_53, ccm_54]

def get_5_filter_true_ccms(images, n_distance, levels = 256):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angles_11 = angles_12 = angles_34 = angles_52 = angles_55 = [45, 135, 225, 315]
    angles_13 = angles_24 = angles_53 = [0, 180]
    angles_14 = angles_23 = angles_54 = [90, 270]
    angles_15 = [0, 90, 180, 270]
    condition = [[lambda x: x, lambda x: x]]

    ccm_all_11 = greycomatrix_channels_extened(images[0], images[0], [2*x for x in n_distance], angles_11, levels=levels, conditions = condition)
    ccm_all_22 = greycomatrix_channels_extened(images[1], images[1], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    ccm_all_33 = greycomatrix_channels_extened(images[2], images[2], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    ccm_all_44 = greycomatrix_channels_extened(images[3], images[3], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    ccm_all_55 = greycomatrix_channels_extened(images[4], images[4], [2*x for x in n_distance], angles_55, levels=levels, conditions = condition)

    ccm_all_12 = greycomatrix_channels_extened(images[0], images[1], [2*x-1 for x in n_distance], angles_12, levels=levels, conditions = condition)
    ccm_all_13 = greycomatrix_channels_extened(images[0], images[2], [2*x-1 for x in n_distance], angles_13, levels=levels, conditions = condition)
    ccm_all_14 = greycomatrix_channels_extened(images[0], images[3], [2*x-1 for x in n_distance], angles_14, levels=levels, conditions = condition)
    ccm_all_15 = greycomatrix_channels_extened(images[0], images[4], [2*x for x in n_distance], angles_15, levels=levels, conditions = condition)
    ccm_all_23 = greycomatrix_channels_extened(images[1], images[2], [2*x-1 for x in n_distance], angles_23, levels=levels, conditions = condition)
    ccm_all_24 = greycomatrix_channels_extened(images[1], images[3], [2*x-1 for x in n_distance], angles_24, levels=levels, conditions = condition)
    ccm_all_52 = greycomatrix_channels_extened(images[4], images[1], [2*x-1 for x in n_distance], angles_52, levels=levels, conditions = condition)
    ccm_all_34 = greycomatrix_channels_extened(images[2], images[3], [2*x-1 for x in n_distance], angles_34, levels=levels, conditions = condition)
    ccm_all_53 = greycomatrix_channels_extened(images[4], images[2], [2*x-1 for x in n_distance], angles_53, levels=levels, conditions = condition)
    ccm_all_54 = greycomatrix_channels_extened(images[4], images[3], [2*x-1 for x in n_distance], angles_54, levels=levels, conditions = condition)

    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    ccm_11 = normalize_over_distances(np.sum(ccm_all_11, axis=3))
    ccm_22 = normalize_over_distances(np.sum(ccm_all_22, axis=3))
    ccm_33 = normalize_over_distances(np.sum(ccm_all_33, axis=3))
    ccm_44 = normalize_over_distances(np.sum(ccm_all_44, axis=3))
    ccm_55 = normalize_over_distances(np.sum(ccm_all_55, axis=3))
    ccm_12 = normalize_over_distances(np.sum(ccm_all_12, axis=3))
    ccm_13 = normalize_over_distances(np.sum(ccm_all_13, axis=3))
    ccm_14 = normalize_over_distances(np.sum(ccm_all_14, axis=3))
    ccm_15 = normalize_over_distances(np.sum(ccm_all_15, axis=3))
    ccm_23 = normalize_over_distances(np.sum(ccm_all_23, axis=3))
    ccm_24 = normalize_over_distances(np.sum(ccm_all_24, axis=3))
    ccm_52 = normalize_over_distances(np.sum(ccm_all_52, axis=3))
    ccm_34 = normalize_over_distances(np.sum(ccm_all_34, axis=3))
    ccm_53 = normalize_over_distances(np.sum(ccm_all_53, axis=3))
    ccm_54 = normalize_over_distances(np.sum(ccm_all_54, axis=3))
    return [ccm_11, ccm_22, ccm_33, ccm_44, ccm_55, ccm_12, ccm_13, ccm_14, ccm_15, ccm_23, ccm_24, ccm_52, ccm_34, ccm_53, ccm_54]

def get_4_filter_cfa_ccms(image, n_distance, levels = 256):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angles_12 = angles_34 = [45, 135, 225, 315]
    angles_13 = angles_24 = [0, 180]
    angles_14 = angles_23 = [90, 270]

    ccm_all_11 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_4[0])
    ccm_all_22 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_4[1])
    ccm_all_33 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_4[2])
    ccm_all_44 = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_4[3])

    ccm_all_12 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_12, levels=levels, conditions = FILTER_CONDITIONS_4[0])
    ccm_all_13 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_13, levels=levels, conditions = FILTER_CONDITIONS_4[0])
    ccm_all_14 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_14, levels=levels, conditions = FILTER_CONDITIONS_4[0])
    ccm_all_23 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_23, levels=levels, conditions = FILTER_CONDITIONS_4[1])
    ccm_all_24 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_24, levels=levels, conditions = FILTER_CONDITIONS_4[1])
    ccm_all_34 = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], angles_34, levels=levels, conditions = FILTER_CONDITIONS_4[2])

    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    ccm_11 = normalize_over_distances(np.sum(ccm_all_11, axis=3))
    ccm_22 = normalize_over_distances(np.sum(ccm_all_22, axis=3))
    ccm_33 = normalize_over_distances(np.sum(ccm_all_33, axis=3))
    ccm_44 = normalize_over_distances(np.sum(ccm_all_44, axis=3))
    ccm_12 = normalize_over_distances(np.sum(ccm_all_12, axis=3))
    ccm_13 = normalize_over_distances(np.sum(ccm_all_13, axis=3))
    ccm_14 = normalize_over_distances(np.sum(ccm_all_14, axis=3))
    ccm_23 = normalize_over_distances(np.sum(ccm_all_23, axis=3))
    ccm_24 = normalize_over_distances(np.sum(ccm_all_24, axis=3))
    ccm_34 = normalize_over_distances(np.sum(ccm_all_34, axis=3))
    return [ccm_11, ccm_22, ccm_33, ccm_44, ccm_12, ccm_13, ccm_14, ccm_23, ccm_24, ccm_34]

def get_4_filter_true_ccms(images, n_distance, levels = 256):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angles_12 = angles_34 = [45, 135, 225, 315]
    angles_13 = angles_24 = [0, 180]
    angles_14 = angles_23 = [90, 270]
    condition = [[lambda x: x, lambda x: x]]

    ccm_all_11 = greycomatrix_channels_extened(images[0], images[0], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    ccm_all_22 = greycomatrix_channels_extened(images[1], images[1], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    ccm_all_33 = greycomatrix_channels_extened(images[2], images[2], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    ccm_all_44 = greycomatrix_channels_extened(images[3], images[3], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)

    ccm_all_12 = greycomatrix_channels_extened(images[0], images[1], [2*x-1 for x in n_distance], angles_12, levels=levels, conditions = condition)
    ccm_all_13 = greycomatrix_channels_extened(images[0], images[2], [2*x-1 for x in n_distance], angles_13, levels=levels, conditions = condition)
    ccm_all_14 = greycomatrix_channels_extened(images[0], images[3], [2*x-1 for x in n_distance], angles_14, levels=levels, conditions = condition)
    ccm_all_23 = greycomatrix_channels_extened(images[1], images[2], [2*x-1 for x in n_distance], angles_23, levels=levels, conditions = condition)
    ccm_all_24 = greycomatrix_channels_extened(images[1], images[3], [2*x-1 for x in n_distance], angles_24, levels=levels, conditions = condition)
    ccm_all_34 = greycomatrix_channels_extened(images[2], images[3], [2*x-1 for x in n_distance], angles_34, levels=levels, conditions = condition)

    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    ccm_11 = normalize_over_distances(np.sum(ccm_all_11, axis=3))
    ccm_22 = normalize_over_distances(np.sum(ccm_all_22, axis=3))
    ccm_33 = normalize_over_distances(np.sum(ccm_all_33, axis=3))
    ccm_44 = normalize_over_distances(np.sum(ccm_all_44, axis=3))
    ccm_12 = normalize_over_distances(np.sum(ccm_all_12, axis=3))
    ccm_13 = normalize_over_distances(np.sum(ccm_all_13, axis=3))
    ccm_14 = normalize_over_distances(np.sum(ccm_all_14, axis=3))
    ccm_23 = normalize_over_distances(np.sum(ccm_all_23, axis=3))
    ccm_24 = normalize_over_distances(np.sum(ccm_all_24, axis=3))
    ccm_34 = normalize_over_distances(np.sum(ccm_all_34, axis=3))
    return [ccm_11, ccm_22, ccm_33, ccm_44, ccm_12, ccm_13, ccm_14, ccm_23, ccm_24, ccm_34]

def get_3_filter_cfa_ccms(image, n_distance, levels = 256):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    rg_angles = bg_angles = [0, 90, 180, 270]
    rb_angles = [45, 135, 225, 315]

    rr_ccm_all = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_3[2])
    gg_ccm_all = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_3[1])
    bb_ccm_all = greycomatrix_channels_extened(image, image, [2*x for x in n_distance], all_angles, levels=levels, conditions = FILTER_CONDITIONS_3[0])

    rg_ccm_all = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], rg_angles, levels=levels, conditions = FILTER_CONDITIONS_3[2])
    bg_ccm_all = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], bg_angles, levels=levels, conditions = FILTER_CONDITIONS_3[0])
    rb_ccm_all = greycomatrix_channels_extened(image, image, [2*x-1 for x in n_distance], rb_angles, levels=levels, conditions = FILTER_CONDITIONS_3[2])

    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    rr_ccm = normalize_over_distances(np.sum(rr_ccm_all, axis=3))
    gg_ccm = normalize_over_distances(np.sum(gg_ccm_all, axis=3))
    bb_ccm = normalize_over_distances(np.sum(bb_ccm_all, axis=3))
    rg_ccm = normalize_over_distances(np.sum(rg_ccm_all, axis=3))
    bg_ccm = normalize_over_distances(np.sum(bg_ccm_all, axis=3))
    rb_ccm = normalize_over_distances(np.sum(rb_ccm_all, axis=3))
    return [rr_ccm, gg_ccm, bb_ccm, rg_ccm, bg_ccm, rb_ccm]

def get_3_filter_true_ccms(images, n_distance, levels = 256):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    rg_angles = bg_angles = [0, 90, 180, 270]
    rb_angles = [45, 135, 225, 315]
    condition = [[lambda x: x, lambda x: x]]

    rr_ccm_all = greycomatrix_channels_extened(images[2], images[2], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    gg_ccm_all = greycomatrix_channels_extened(images[1], images[1], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)
    bb_ccm_all = greycomatrix_channels_extened(images[0], images[0], [2*x for x in n_distance], all_angles, levels=levels, conditions = condition)

    rg_ccm_all = greycomatrix_channels_extened(images[2], images[1], [2*x-1 for x in n_distance], rg_angles, levels=levels, conditions = condition)
    bg_ccm_all = greycomatrix_channels_extened(images[0], images[1], [2*x-1 for x in n_distance], bg_angles, levels=levels, conditions = condition)
    rb_ccm_all = greycomatrix_channels_extened(images[2], images[0], [2*x-1 for x in n_distance], rb_angles, levels=levels, conditions = condition)
    
    # neighbourhood should be identical, that's why I commented it out
    """
    rr_ccm_all = greycomatrix_channels_extened(images[2], images[2], n_distance, all_angles, levels=levels, conditions = condition)
    gg_ccm_all = greycomatrix_channels_extened(images[1], images[1], n_distance, all_angles, levels=levels, conditions = condition)
    bb_ccm_all = greycomatrix_channels_extened(images[0], images[0], n_distance, all_angles, levels=levels, conditions = condition)

    rg_ccm_all = greycomatrix_channels_extened(images[2], images[1], n_distance, all_angles, levels=levels, conditions = condition)
    bg_ccm_all = greycomatrix_channels_extened(images[0], images[1], n_distance, all_angles, levels=levels, conditions = condition)
    rb_ccm_all = greycomatrix_channels_extened(images[2], images[0], n_distance, all_angles, levels=levels, conditions = condition)
    """

    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    rr_ccm = normalize_over_distances(np.sum(rr_ccm_all, axis=3))
    gg_ccm = normalize_over_distances(np.sum(gg_ccm_all, axis=3))
    bb_ccm = normalize_over_distances(np.sum(bb_ccm_all, axis=3))
    rg_ccm = normalize_over_distances(np.sum(rg_ccm_all, axis=3))
    bg_ccm = normalize_over_distances(np.sum(bg_ccm_all, axis=3))
    rb_ccm = normalize_over_distances(np.sum(rb_ccm_all, axis=3))
    return [rr_ccm, gg_ccm, bb_ccm, rg_ccm, bg_ccm, rb_ccm]

def normalize_over_distances(ccm):
    ccm = ccm.astype(np.float32)
    for idx in range(len(CCM_DISTANCES)):
        ccm[:, :, idx] = normalize_ccm(ccm[:, :, idx])
    return ccm

def normalize_ccm(ccm_specific_distance):
    normalized_ccm = ccm_specific_distance/np.sum(ccm_specific_distance, axis=(0, 1))
    trial = normalized_ccm.sum()
    return normalized_ccm

def qunatization(img, quant):
    return (img // ((MAX_PIXEL_VALUE + 1) / quant)).astype(np.uint8)

def get_filter_and_cfa_channels(sample):
    filter_channels = []
    for root, _, fnames in sorted(os.walk(NEW_DATASET_DIR + sample)):
        for fname in fnames:
            if is_image_file(fname, file_extensions=IMG_FILE_EXTENSIONS):
                if "cfa" in fname:
                    cfa_image = cv2.imread(root + "/" + fname, cv2.IMREAD_GRAYSCALE)
                else:
                    filter_channels.append(cv2.imread(root + "/" + fname, cv2.IMREAD_GRAYSCALE))
    return cfa_image, filter_channels

def get_filter_and_cfa_crops(cfa_image, filter_channels, subsample_crop):
    cfa_crop = cfa_image[subsample_crop[0]:subsample_crop[0] + subsample_crop[2], subsample_crop[1]:subsample_crop[1] + subsample_crop[3]]
    filters_crop = [filter_channel[subsample_crop[0]:subsample_crop[0] + subsample_crop[2], subsample_crop[1]:subsample_crop[1] + subsample_crop[3]] for filter_channel in filter_channels]
    return cfa_crop,filters_crop

def get_filter_and_cfa_quantization(quant, cfa_crop, filters_crop):
    cfa_crop_quant = qunatization(cfa_crop, quant)
    filters_crop_quant = [qunatization(filter_crop, quant) for filter_crop in filters_crop]
    return cfa_crop_quant,filters_crop_quant

def calculate_crops_ccms(cfa_image, filter_channels, subsamples_crops, quant):
    cfa_ccms = []
    true_ccms = []
    for subsample_crop in subsamples_crops:
        cfa_crop, filters_crop = get_filter_and_cfa_crops(cfa_image, filter_channels, subsample_crop)

        cfa_crop_quant, filters_crop_quant = get_filter_and_cfa_quantization(quant, cfa_crop, filters_crop)

        #start_time = time.time()
        cfa_ccms.append(get_5_filter_cfa_ccms(cfa_crop_quant, n_distance=CCM_DISTANCES, levels=quant))
        #print("--- CFA time: %s seconds ---" % (time.time() - start_time))
        #start_time = time.time()
        true_ccms.append(get_5_filter_true_ccms(filters_crop_quant, n_distance=CCM_DISTANCES, levels=quant))
        #print("--- True list time: %s seconds ---" % (time.time() - start_time))
    return cfa_ccms, true_ccms

def quantize_crops(sample, cfa_image, filter_channels, subsamples_crops):
    for quant in QUANTIZATION:
        print("Sample {} , quant {} started".format(sample, quant))
        cfa_ccms, true_ccms = calculate_crops_ccms(cfa_image, filter_channels, subsamples_crops, quant)
        save_obj(cfa_ccms, NEW_DATASET_DIR + sample + "/cfa_ccms_{}.pkl".format(quant))
        save_obj(true_ccms, NEW_DATASET_DIR + sample + "/true_ccms_{}.pkl".format(quant))

def sample_loop(sample, subsample_dict):
    cfa_image, filter_channels = get_filter_and_cfa_channels(sample)
    subsamples_crops = subsample_dict[sample]
    start_time = time.time()
    quantize_crops(sample, cfa_image, filter_channels, subsamples_crops)
    print("--- {} is done: {} seconds ---".format(sample, (time.time() - start_time)))


def generate_ccms():
    subsample_dict = load_obj(GENERAL_DATASET_DIR + SUBSAMPLE_DICT)
    samples = [sample for sample in subsample_dict]
    print("Number of processors: ", mp.cpu_count())
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        results = pool.map(partial(sample_loop, subsample_dict = subsample_dict), samples)
    pool.close()

if __name__ == "__main__":
    # FIRST STEP
    # create subsample dictionary
    #subsample_images(SUBSAMPLE_DICT)

    # SECOND STEP
    # get all image paths
    #img_paths = get_img_paths(HYTEXILA_DIR, NEW_DATASET_DIR)
    # create dataset
    #create_channels_dataset(img_paths)

    # THIRD STEP
    # generate CFA images
    #generate_cfa_images(NEW_DATASET_DIR, get_cfa_image, FILTER_CONDITIONS)

    # generate CCMs
    start_time = time.time()
    generate_ccms()
    print("--- Time needed for 10 processes: %s seconds ---" % (time.time() - start_time))

    # example for one image
    #one_image_example()

    