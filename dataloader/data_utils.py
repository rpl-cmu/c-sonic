import os
from pathlib import Path
from random import shuffle

import cv2
import numpy as np
import torch
import torch.utils.data
from scipy.spatial import cKDTree

from .demo_superpoint import SuperPointFrontend

'''Minor modifications from CAPS'''

REPO_ROOT = Path(__file__).resolve().parents[1]

# One SuperPoint frontend per weights file, so every image does not reload it.
_superpoint_cache = {}


def _get_superpoint(weights_path=None):
    """Return the SuperPoint frontend, building it on first use.

    The weights are looked up in order: the path given, the same path relative to
    the working directory, then the copy in the repository.
    """
    candidates = [weights_path, 'pretrained/superpoint_v1.pth',
                  str(REPO_ROOT / 'pretrained' / 'superpoint_v1.pth')]
    resolved = next((os.path.abspath(c) for c in candidates
                     if c and os.path.isfile(c)), None)
    if resolved is None:
        raise FileNotFoundError(
            'SuperPoint weights not found; download pretrained/superpoint_v1.pth '
            'as described in README.md')
    # A dataloader worker keeps the frontend on the CPU: the CUDA context does
    # not survive the fork that starts the worker. Ask about the worker first:
    # by then the parent has initialised CUDA, and querying it in a forked child
    # is exactly what this guard exists to avoid.
    use_cuda = torch.utils.data.get_worker_info() is None and torch.cuda.is_available()
    key = (resolved, use_cuda)
    if key not in _superpoint_cache:
        _superpoint_cache[key] = SuperPointFrontend(
            weights_path=resolved, nms_dist=4, conf_thresh=0.015, nn_thresh=0.7,
            cuda=use_cuda)
    return _superpoint_cache[key]


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=3.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def rescale_keypoints(keypoints, size):
    """ Rescale keypoints to fit original image size.
    Inputs
      keypoints: Nx2 numpy array of keypoints.
      size: (H, W) tuple specifying original image size.
    Returns
      rescaled_keypoints: Nx2 numpy array of rescaled keypoints.
    """
    H, W = size
    rescaled_keypoints = keypoints.copy()
    rescaled_keypoints[:, 0] = keypoints[:, 0] * W / 640
    rescaled_keypoints[:, 1] = keypoints[:, 1] * H / 480
    return rescaled_keypoints

def preprocess_image(image, size):
    """ Preprocess image before generating keypoints for both akaze (norm_image) and superpoint(grayim)"""
    norm_image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    norm_image = unsharp_mask(norm_image, kernel_size=(3, 3), sigma=0.6, amount=2.0, threshold=0)
    norm_image = norm_image.astype(np.uint8)

    # set up image for superpoint
    interp= cv2.INTER_AREA
    grayim = cv2.resize(norm_image, (640,480), interpolation=interp)
    grayim = grayim.astype(np.float32) / 255.

    return norm_image, grayim


def generate_query_kpts(img, num_pts, min_kpts, h, w, superpoint_weights=None):
    """Detect query keypoints with AKAZE and SuperPoint.

    Returns ``num_pts`` x 2 coordinates. When either detector finds fewer than
    ``min_kpts`` points the image is too poor to train on, so a single zero point
    is returned instead.
    """
    superpoint = _get_superpoint(superpoint_weights)
    akaze = cv2.AKAZE_create()

    norm_image, grayim = preprocess_image(img, (h, w))

    kpa = akaze.detect(norm_image, None)
    kps, _, _ = superpoint.run(grayim)
    rescaled_kpts = rescale_keypoints(kps.T, (h, w))
    kpsp = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in rescaled_kpts]

    if len(kpa) < min_kpts or len(kpsp) < min_kpts:
        return np.zeros((1, 2))

    coord_a = np.array([[kp.pt[0], kp.pt[1]] for kp in kpa])
    coord_s = np.array([[kp.pt[0], kp.pt[1]] for kp in kpsp])
    coord = np.vstack((coord_a, coord_s))

    if len(coord) > num_pts:
        # Keep a random subset of the detections.
        return coord[np.random.choice(coord.shape[0], num_pts, replace=False), :]
    return _fill_with_neighbors(coord, num_pts)


def _fill_with_neighbors(coord, num_pts, rng=np.random):
    """Pad ``coord`` up to ``num_pts`` rows.

    Each new point is the mean of a detected point and its three nearest
    neighbours, so the padding stays inside the detected region.
    """
    total = len(coord)
    if total >= num_pts:
        return coord

    _, idx = cKDTree(coord).query(coord, k=min(4, total))
    idx = idx.reshape(total, -1)
    points = rng.randint(low=0, high=total - 1, size=num_pts - total)
    gen_kpts = [[np.mean(coord[idx[i], 0]), np.mean(coord[idx[i], 1])] for i in points]
    shuffle(gen_kpts)

    return np.vstack((coord, gen_kpts[:(num_pts - total)]))
