import matplotlib.pyplot as plt
from skimage.feature import greycomatrix
import numpy as np
from skimage.feature.texture import greycomatrix_channels
from PIL import Image
import time

def display_angle(specific_distance_mat, levels):
    return specific_distance_mat.flatten().reshape(levels, levels)

def get_ccms(image, n_distance, levels = 255):
    # Angles are going around clockwise!!! 0 degrees is pixel to the right, 90 degrees is pixel below etc.
    all_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    rg_angles = bg_angles = [0, 90, 180, 270]
    rb_angles = [45, 135, 225, 315]

    start_time = time.time()
    rr_ccm_all = greycomatrix_channels(image, image, [2*x for x in n_distance], all_angles, levels=levels, channels='r')
    print("--- RR: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    gg_ccm_all = greycomatrix_channels(image, image, [2*x for x in n_distance], all_angles, levels=levels, channels='g')
    print("--- GG: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    bb_ccm_all = greycomatrix_channels(image, image, [2*x for x in n_distance], all_angles, levels=levels, channels='b')
    print("--- BB: %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    rg_ccm_all = greycomatrix_channels(image, image, [2*x-1 for x in n_distance], rg_angles, levels=levels, channels='r')
    print("--- RG: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    bg_ccm_all = greycomatrix_channels(image, image, [2*x-1 for x in n_distance], bg_angles, levels=levels, channels='b')
    print("--- BG: %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    rb_ccm_all = greycomatrix_channels(image, image, [2*x-1 for x in n_distance], rb_angles, levels=levels, channels='r')
    print("--- RB: %s seconds ---" % (time.time() - start_time))
    
    # before - four dimensions x, y, distances, angles
    # summed angles - only three dimensions now: x, y, distances
    rr_ccm = np.sum(rr_ccm_all, axis=3)
    gg_ccm = np.sum(gg_ccm_all, axis=3)
    bb_ccm = np.sum(bb_ccm_all, axis=3)
    rg_ccm = np.sum(rg_ccm_all, axis=3)
    bg_ccm = np.sum(bg_ccm_all, axis=3)
    rb_ccm = np.sum(rb_ccm_all, axis=3)
    return rr_ccm, gg_ccm, bb_ccm, rg_ccm, bg_ccm, rb_ccm

def normalize_ccm(ccm_specific_distance):
    normalized_ccm = ccm_specific_distance/np.sum(ccm_specific_distance, axis=(0, 1))
    return normalized_ccm

if __name__ == "__main__":
    image = np.array([[0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 2, 2, 2],
                   [2, 2, 3, 3]], dtype=np.uint8)
    levels = 4

    image = Image.open('textile_patch.jpg').convert('L')
    levels = 255
    # distance d = 1
    rr_ccm, gg_ccm, bb_ccm, rg_ccm, bg_ccm, rb_ccm = get_ccms(image, [1], levels=levels)

    start_time = time.time()
    rr_ccm_normalized = normalize_ccm(rr_ccm[:, :, 0])
    print("--- Normalization: %s seconds ---" % (time.time() - start_time))
    print("\nRed-Red CCM with n = 1:")
    print(display_angle(rr_ccm[:, :, 0], levels))
    print("\nSum of all occurrences in Red-Red CCM")
    print(np.sum(rr_ccm[:, :, 0], axis=(0, 1)))
    print("\nNormalized Red-Red CCM with n = 1:")