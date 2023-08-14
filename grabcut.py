import numpy as np
import igraph as ig
import cv2
import time
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel
mask_global = []

# grabcut algorithm
def grabcut(img, rect, n_iter=5):
    global mask_global
    previous_energy = 0
    img = np.asarray(img, dtype=np.float64)

    # initialize
    mask_global = np.zeros(img.shape[:2], dtype=np.uint8)
    mask_global.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    # Initalize the small rectangle Foreground
    mask_global[y:y + h, x:x + w] = GC_PR_FGD
    mask_global[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask_global)

    num_iters = 1000
    for i in range(num_iters):
        # GMM update

        bgGMM, fgGMM = update_GMMs(img, mask_global, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask_global, bgGMM, fgGMM)

        print("Iteration: " + str(i + 1) + " after calculating mincut, energy is: " + str(energy))
        mask_global = update_mask(mincut_sets, mask_global)
        print("difference between energies: " + str(np.abs(energy - previous_energy)))
        if check_convergence(np.abs(energy - previous_energy)):
            print("iterations to convergence: " + str(i + 1))
            break
        previous_energy = energy

    # Return mask&GMMs
    return mask_global, bgGMM, fgGMM


# initialize GMM
def initalize_GMMs(img, mask, n_components=5):
    # Extract fg&bg pixels from mask
    bg_pix = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)
    fg_pix = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)

    # Initialize using k-means clustering algorithm & random state 2
    kmeans_bg = KMeans(n_clusters=n_components, random_state=2)
    kmeans_bg.fit(bg_pix)
    kmeans_fg = KMeans(n_clusters=n_components, random_state=2)
    kmeans_fg.fit(fg_pix)

    # Initialize GMMs using k-means.cluster_centers algorithm & random state 2
    fgGMM = GaussianMixture(n_components=n_components, means_init=kmeans_fg.cluster_centers_, random_state=2)
    fgGMM.fit(fg_pix)
    bgGMM = GaussianMixture(n_components=n_components, means_init=kmeans_bg.cluster_centers_, random_state=2)
    bgGMM.fit(bg_pix)

    return bgGMM, fgGMM


# update GMM using img, mask, bgGMM, fgGMM
def update_GMMs(img, mask, bgGMM, fgGMM):
    fg_pix = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)].reshape(-1, 3)
    bg_pix = img[np.logical_or(mask == GC_PR_BGD, mask == GC_BGD)].reshape(-1, 3)

    # init bgGMM
    bg_components = bgGMM.n_components
    bg_weights = np.zeros(bg_components)
    bg_means = np.zeros((bg_components, 3))
    bg_covs = np.zeros((bg_components, 3, 3))

    # calc mean&covariance for all GMM components and update the GMM data (weights, mean and cov)
    for i in range(bg_components):
        component_mask = bgGMM.predict(bg_pix) == i
        component_data = bg_pix[component_mask]

        if len(component_data) > 0:
            bg_weights[i] = len(component_data) / len(bg_pix)
            cov, mean = cv2.calcCovarMatrix(component_data, None, cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            bg_means[i] = mean.flatten()
            bg_covs[i] = cov

    # Update bgGMM data
    bgGMM.means_ = bg_means
    bgGMM.weights_ = bg_weights
    bgGMM.covariances_ = bg_covs

    # Update fgGMM
    fg_components = fgGMM.n_components
    fg_weights = np.zeros(fg_components)
    fg_means = np.zeros((fg_components, 3))
    fg_covs = np.zeros((fg_components, 3, 3))

    for i in range(fg_components):
        component_mask = fgGMM.predict(fg_pix) == i
        component_data = fg_pix[component_mask]

        if len(component_data) > 0:
            fg_weights[i] = len(component_data) / len(fg_pix)
            cov, mean = cv2.calcCovarMatrix(component_data, None, cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            fg_means[i] = mean.flatten()
            fg_covs[i] = cov

    fgGMM.means_ = fg_means
    fgGMM.weights_ = fg_weights
    fgGMM.covariances_ = fg_covs

    # remove components with the weight of 0
    fg_index_list = []
    bg_index_list = []

    for i in range(len(fgGMM.weights_)):
        if fgGMM.weights_[i] <= 0.005:
            fg_index_list.append(i)

    for i in range(len(bgGMM.weights_)):
        if bgGMM.weights_[i] <= 0.005:
            bg_index_list.append(i)

    if len(bg_index_list) > 0:
        bgGMM.n_components = bgGMM.n_components - len(bg_index_list)
        bgGMM.weights_ = np.delete(bgGMM.weights_, bg_index_list, axis=0)
        bgGMM.precisions_ = np.delete(bgGMM.precisions_, bg_index_list, axis=0)
        bgGMM.precisions_cholesky_ = np.delete(bgGMM.precisions_cholesky_, bg_index_list, axis=0)
        bgGMM.means_ = np.delete(bgGMM.means_, bg_index_list, axis=0)
        bgGMM.covariances_ = np.delete(bgGMM.covariances_, bg_index_list, axis=0)
        bgGMM.means_init = np.delete(bgGMM.means_init, bg_index_list, axis=0)

    if len(fg_index_list) > 0:
        fgGMM.n_components = fgGMM.n_components - len(fg_index_list)
        fgGMM.weights_ = np.delete(fgGMM.weights_, fg_index_list, axis=0)
        fgGMM.precisions_ = np.delete(fgGMM.precisions_, fg_index_list, axis=0)
        fgGMM.precisions_cholesky_ = np.delete(fgGMM.precisions_cholesky_, fg_index_list, axis=0)
        fgGMM.means_ = np.delete(fgGMM.means_, fg_index_list, axis=0)
        fgGMM.covariances_ = np.delete(fgGMM.covariances_, fg_index_list, axis=0)
        fgGMM.means_init = np.delete(fgGMM.means_init, fg_index_list, axis=0)

    print("bgGMM weights: " + str(bgGMM.weights_))
    print("fgGMM weights: " + str(fgGMM.weights_))
    return bgGMM, fgGMM


# calculate mincut function
def calculate_mincut(img, mask, bgGMM, fgGMM):
    min_cut = [[], []]
    energy = 0
    h, w = img.shape[:2]
    img_indices = np.arange(h * w).reshape(h, w)
    graph = ig.Graph()
    graph.add_vertices(h * w + 2)
    source = h * w
    sink = h * w + 1

    # Calculate the D for the t-link as described in the article
    d_val = np.zeros((h, w, 2))
    img_reshaped = img.reshape(h * w, -1)
    bg_pix = -bgGMM.score_samples(img_reshaped).reshape(h, w, 1)
    fg_pix = -fgGMM.score_samples(img_reshaped).reshape(h, w, 1)
    d_val = np.concatenate((bg_pix, fg_pix), axis=2)
    gamma = 50

    # calculate beta as described in the article
    beta = 0
    for row in range(h):
        for col in range(w):
            # calculate euclidean distance&sum
            if row < h - 1:
                beta += np.linalg.norm(img[row, col] - img[row + 1, col]) ** 2
            if row > 0:
                beta += np.linalg.norm(img[row, col] - img[row - 1, col]) ** 2
            if col > 0:
                beta += np.linalg.norm(img[row, col] - img[row, col - 1]) ** 2
            if col < w - 1:
                beta += np.linalg.norm(img[row, col] - img[row, col + 1]) ** 2
            if row > 0 and col > 0:
                beta += np.linalg.norm(img[row, col] - img[row - 1, col - 1]) ** 2
            if row > 0 and col < w - 1:
                beta += np.linalg.norm(img[row, col] - img[row - 1, col + 1]) ** 2
            if row < h - 1 and col < w - 1:
                beta += np.linalg.norm(img[row, col] - img[row + 1, col + 1]) ** 2
            if row < h - 1 and col > 0:
                beta += np.linalg.norm(img[row, col] - img[row + 1, col - 1]) ** 2

    # calculate number of pixels
    num_pixels = h * w
    beta = beta / ((8 * num_pixels) - (2 * h + 2 * w))
    beta = 2 * beta
    beta = 1 / beta
    # end of calculating the beta

    edges = []
    weights = []

    # calc N-link as described in the article
    for row in range(h):
        for col in range(w):
            # save K (the maximum of all N-links)
            k = 0
            node1 = img_indices[row, col]
            # edge weight of top neighbor
            if row > 0:
                node2 = img_indices[row - 1, col]
                # color difference calc
                color_diff = (np.linalg.norm(img[row, col] - img[row - 1, col])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of left neighbor
            if col > 0:
                node2 = img_indices[row, col - 1]
                color_diff = (np.linalg.norm(img[row, col] - img[row, col - 1])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of bottom neighbor
            if row < h - 1:
                node2 = img_indices[row + 1, col]
                color_diff = (np.linalg.norm(img[row, col] - img[row + 1, col])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of right neighbor
            if col < w - 1:
                node2 = img_indices[row, col + 1]
                color_diff = (np.linalg.norm(img[row, col] - img[row, col + 1])) ** 2
                edge_weight = gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of left neighbor
            if row > 0 and col > 0:
                node2 = img_indices[row - 1, col - 1]
                color_diff = (np.linalg.norm(img[row, col] - img[row - 1, col - 1])) ** 2
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of top right neighbor
            if row > 0 and col < w - 1:
                node2 = img_indices[row - 1, col + 1]
                color_diff = (np.linalg.norm(img[row, col] - img[row - 1, col + 1])) ** 2
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of bottom left neighbor
            if row < h - 1 and col > 0:
                node2 = img_indices[row + 1, col - 1]
                color_diff = (np.linalg.norm(img[row, col] - img[row + 1, col - 1])) ** 2
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # edge weight of bottom right neighbor
            if row < h - 1 and col < w - 1:
                node2 = img_indices[row + 1, col + 1]
                color_diff = (np.linalg.norm(img[row, col] - img[row + 1, col + 1])) ** 2
                edge_weight = (1 / np.sqrt(2)) * gamma * np.exp(-beta * color_diff)
                edges.append((node1, node2))
                weights.append(edge_weight)
                if edge_weight > k:
                    k = edge_weight

            # calculate D according to the bg
            Dbg = d_val[row, col, 0]
            # calculate D according to the fg
            Dfg = d_val[row, col, 1]
            edges.append((source, node1))
            edges.append((node1, sink))

            # put the weight for the t-link according to the status of the pixel (hardbg, hard fg, soft..)
            # according to the table in the article
            if mask[row, col] == GC_PR_FGD or mask[row, col] == GC_PR_BGD:
                weights.append(Dfg)
                weights.append(Dbg)
            if mask[row, col] == GC_FGD:
                weights.append(0)
                weights.append(k)
            if mask[row, col] == GC_BGD:
                weights.append(k)
                weights.append(0)

    graph.add_edges(edges, {'weight': weights})
    mincut = graph.st_mincut(source, sink, capacity='weight')
    mincut_sets = [set(mincut.partition[0]), set(mincut.partition[1])]

    # return the partition and the energy
    return mincut_sets, mincut.value

# update mask function
def update_mask(mincut_sets, mask):
    h, w = mask.shape
    img_indices = np.arange(h * w).reshape(h, w)
    new_mask = np.copy(mask)
    for row in range(h):
        for col in range(w):
            if img_indices[row, col] in mincut_sets[0] and (mask[row, col] == GC_PR_BGD or mask[row, col] == GC_PR_FGD):
                new_mask[row, col] = GC_PR_BGD
            elif img_indices[row, col] in mincut_sets[1] and (mask[row, col] == GC_PR_BGD or mask[row, col] == GC_PR_FGD):
                new_mask[row, col] = GC_PR_FGD
    mask = np.copy(new_mask)

    # return the updated mask
    return mask


# check convergence function
def check_convergence(energy):
    global mask_global
    convergence = False
    if energy <= 1750:
        # change soft bg pixels to (hard) bg pixels
        mask_global[mask_global == GC_PR_BGD] = GC_BGD
        convergence = True
    # return the convergence value
    return convergence


def cal_metric(predicted_mask, gt_mask):
    predicted_mask_bool = predicted_mask.astype(bool)
    gt_mask_bool = gt_mask.astype(bool)

    # Calc the num of the pixels the labeled correctly
    correct_pixels = np.sum(predicted_mask_bool == gt_mask_bool)
    total_pixels = predicted_mask.size

    # Jaccard
    intersection = np.sum(predicted_mask_bool & gt_mask_bool)
    union = np.sum(predicted_mask_bool | gt_mask_bool)
    jaccard_similarity = intersection / union

    # Accuracy
    accuracy = correct_pixels / total_pixels

    return accuracy, jaccard_similarity


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='bush', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)
    # add this line for high blur: img = cv2.blur(img, (20, 20))
    # add this line for low blur: img = cv2.blur(img, (5, 5))

    # Run the GrabCut algorithm on the image and bounding box
    mask_global, bgGMM, fgGMM = grabcut(img, rect)
    mask_global = cv2.threshold(mask_global, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask_global, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask_global[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask_global)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

