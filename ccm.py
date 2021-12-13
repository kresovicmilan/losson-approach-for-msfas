import numpy as np
import warnings

def greycomatrix_channels_extened(image1, image2, distances, angles, conditions, levels=None, symmetric=False,
                 normed=False):

    image1 = np.ascontiguousarray(image1)
    image2 = np.ascontiguousarray(image2)

    image_max1 = image1.max()
    image_max2 = image2.max()

    if np.issubdtype(image1.dtype, np.floating):
        raise ValueError("Float images are not supported by greycomatrix. "
                         "Convert the image to an unsigned integer type.")
    
    if np.issubdtype(image2.dtype, np.floating):
        raise ValueError("Float images are not supported by greycomatrix. "
                         "Convert the image to an unsigned integer type.")

    # for image type > 8bit, levels must be set.
    if image1.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError("The levels argument is required for data types "
                         "other than uint8. The resulting matrix will be at "
                         "least levels ** 2 in size.")
    
    # for image type > 8bit, levels must be set.
    if image2.dtype not in (np.uint8, np.int8) and levels is None:
        raise ValueError("The levels argument is required for data types "
                         "other than uint8. The resulting matrix will be at "
                         "least levels ** 2 in size.")

    if np.issubdtype(image1.dtype, np.signedinteger) and np.any(image1 < 0):
        raise ValueError("Negative-valued images are not supported.")
    
    if np.issubdtype(image2.dtype, np.signedinteger) and np.any(image2 < 0):
        raise ValueError("Negative-valued images are not supported.")

    if levels is None:
        levels = 256

    if image_max1 >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")
    
    if image_max2 >= levels:
        raise ValueError("The maximum grayscale value in the image should be "
                         "smaller than the number of levels.")

    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)

    # count co-occurences
    P = glcm_loop_channels_extended(image1, image2, distances, angles, levels, conditions)

    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt

    # normalize each GLCM
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P

def glcm_loop_channels_extended(image1, image2, distances, angles, levels, conditions):
    P = np.zeros((levels, levels, len(distances), len(angles)),dtype=np.uint32, order='C')
    rows = image1.shape[0]
    cols = image1.shape[1]

    # go through all angles
    for a_idx in range(angles.shape[0]):
        angle = angles[a_idx]
        # go through all distances
        for d_idx in range(distances.shape[0]):
            distance = distances[d_idx]
            # making sure that angles of n*45degrees are vertices of square and not in points on circle line
            if int(angle) % 45 == 0 and (int(angle) // 45) % 2 != 0:
                distance = distance*np.sqrt(2)
            # calculate the offset, start and ending point
            offset_row = int(round(np.sin(angle*np.pi/180) * distance))
            offset_col = int(round(np.cos(angle*np.pi/180) * distance))
            start_row = max(0, -offset_row)
            end_row = min(rows, rows - offset_row)
            start_col = max(0, -offset_col)
            end_col = min(cols, cols - offset_col)
            for cond in conditions:
                row_range = [x for x in list(range(start_row, end_row)) if cond[0](x)]
                col_range = [x for x in list(range(start_col, end_col)) if cond[1](x)]
                P = update_ccm(image1, image2, levels, P, a_idx, d_idx, offset_row, offset_col, row_range, col_range)
    
    return P

def update_ccm(image1, image2, levels, P, a_idx, d_idx, offset_row, offset_col, row_range, col_range):
    for r in row_range:
        for c in col_range:
            i = image1[r, c]
            # compute the location of the offset pixel
            row = r + offset_row
            col = c + offset_col
            j = image2[row, col]
            if 0 <= i < levels and 0 <= j < levels:
                P[i, j, d_idx, a_idx] += 1
    return P