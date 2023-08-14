import cv2
from scipy.sparse.linalg import spsolve
import numpy as np
import argparse
import math
import scipy.sparse as sp

def poisson_blend(im_src, im_tgt, im_mask, center):
    # Step 1: Calculate the Laplacian operator

    # dimensions of mask
    mask_height, mask_width = im_mask.shape
    # dimensions of target
    tgt_h, tgt_w, _ = im_tgt.shape

    # (top_left_x, top_left_y) - top left
    # (bottom_right_x, bottom_right_y) - bottom right
    top_left_y, bottom_right_y = center[1] - mask_height // 2, math.ceil(center[1] + mask_height / 2)
    top_left_x, bottom_right_x = center[0] - mask_width // 2, math.ceil(center[0] + mask_width / 2)

    # matrix d clac
    mat_d = create_d(mask_width)

    # matrix a calc using matrix d
    mat_a = sp.block_diag([mat_d] * mask_height).tolil()
    mat_a.setdiag(-1, 1 * mask_width)
    mat_a.setdiag(-1, -1 * mask_width)

    # laplacian matrix calc using matrix a
    laplacian = mat_a.tocsc()

    # Step 2: Solve the Poisson equation
    # Iterate over each pixel in the image mask
    for row in range(1, mask_height - 1):
        for col in range(1, mask_width - 1):
            # If the pixel is part of the background (value of 0 in the mask)
            if im_mask[row, col] == 0:
                # Convert the row and column indices to a single index for the matrix
                matrix_index = col + row * mask_width

                # Set the diagonal element of the matrix to 1
                mat_a[matrix_index, matrix_index] = 1

                # Set the adjacent elements of the matrix to 0
                mat_a[matrix_index, matrix_index + 1] = 0
                mat_a[matrix_index, matrix_index - 1] = 0
                mat_a[matrix_index, matrix_index + mask_width] = 0
                mat_a[matrix_index, matrix_index - mask_width] = 0

    mat_b = mat_a.tocsc()

    # create image for the result
    blended_image = np.copy(im_tgt)

    mask_flat = im_mask.flatten()

    # Step 3: Blend the source and target images using the Poisson result
    for channel in range(im_src.shape[2]):
        source_flat = im_src[:, :, channel].flatten()
        target_flat = im_tgt[top_left_y:bottom_right_y, top_left_x:bottom_right_x, channel].flatten()
        vec_b = laplacian.dot(source_flat)
        vec_b[mask_flat == 0] = target_flat[mask_flat == 0]
        x = spsolve(mat_b, vec_b)
        x = np.clip(x, 0, 255).astype(np.uint8)
        x = x.reshape(mask_height, mask_width)

        blended_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x, channel] = np.where(im_mask == 255, x, im_tgt[top_left_y:bottom_right_y, top_left_x:bottom_right_x, channel])

    # Return the blended image
    return blended_image


def create_d(mask_width):
    mat_d1 = sp.lil_matrix((mask_width, mask_width))
    mat_d1.setdiag(-1, -1)
    mat_d1.setdiag(4)
    mat_d1.setdiag(-1, 1)
    return mat_d1


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/llama.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/llama.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/grass_mountains.jpeg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

