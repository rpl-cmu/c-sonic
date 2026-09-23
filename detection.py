"""Read sonar images and detect the query keypoints the network matches.

This is the inference-time counterpart of ``dataloader/data_utils.py``. The
dataloader samples a fixed number of training points; here every detection is
kept, together with its descriptor, and the images come from files rather than
from a pair list.

Run ``CSONIC_test.py`` to match a pair. This module only prepares its inputs.
"""
import os

import cv2
import numpy as np

from dataloader.data_utils import _get_superpoint, rescale_keypoints, unsharp_mask

__all__ = ['read_meta', 'load_image', 'load_data', 'unsharp_mask',
           'anisotropic_diffusion', 'preprocess_image', 'generate_query_kpts',
           'rescale_keypoints']


def read_meta(path):
    """Return the sonar metadata stored in a ``_meta.npy`` file.

    The dict holds ``width`` and ``height`` in pixels, ``r_min`` and ``r_max``
    in metres, and ``elev`` and ``azi`` in degrees, for example
    ``{'width': 512, 'height': 512, 'r_min': 0.1, 'r_max': 7, 'elev': 12,
    'azi': 60}``.
    """
    return np.load(path, allow_pickle=True).item()


def load_image(path, meta=None):
    """Read one sonar image and put it in the layout the network expects.

    A ``.png`` is read as 8-bit greyscale. A ``.npy`` holds raw intensities, so
    it is scaled to the 8-bit range by its own minimum and maximum.

    The image is then flipped on both axes, as ``SonarData.__getitem__`` does:
    the sensor origin sits at the top centre of the stored image, and the model
    expects it at the bottom centre. When ``meta`` is given and the image does
    not already have the size the metadata states, it is resized to it. The
    real-world PNGs are taller than their metadata says, so this step matters.

    :param path: the image file
    :param meta: the metadata dict from :func:`read_meta`, or None
    :return: a uint8 array of shape (height, width)
    """
    if str(path).endswith('.npy'):
        raw = np.load(path)
        low, high = float(np.min(raw)), float(np.max(raw))
        if high > low:
            image = ((raw - low) / (high - low) * 255).astype(np.uint8)
        else:
            # A constant image has no range to stretch.
            image = np.zeros(raw.shape, dtype=np.uint8)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError('could not read the image: {}'.format(path))

    image = np.flip(image).copy()

    if meta is not None:
        size = (int(meta['width']), int(meta['height']))
        if image.shape != (size[1], size[0]):
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    return image.astype(np.uint8)


def load_data(img1_path, img2_path, pose1_path=None, pose2_path=None,
              meta1_path=None, meta2_path=None):
    """Read an image pair, and its poses and metadata when they are given.

    :return: a dict with ``im1``, ``im2``, ``pose1``, ``pose2``, ``meta1`` and
        ``meta2``. The entries with no path are None. Each pose is the 4x4
        sensor-to-world matrix stored next to the image.
    """
    meta1 = read_meta(meta1_path) if meta1_path is not None else None
    meta2 = read_meta(meta2_path) if meta2_path is not None else None
    return {
        'im1': load_image(img1_path, meta1),
        'im2': load_image(img2_path, meta2),
        'pose1': np.load(pose1_path) if pose1_path is not None else None,
        'pose2': np.load(pose2_path) if pose2_path is not None else None,
        'meta1': meta1,
        'meta2': meta2,
    }


def anisotropic_diffusion(image, iterations=10, delta_t=0.25, kappa=10):
    """Smooth an image while keeping its edges (Perona-Malik diffusion).

    :param image: a greyscale image
    :param iterations: number of diffusion steps
    :param delta_t: the time step
    :param kappa: the edge threshold; a larger value smooths across more edges
    :return: a uint8 image of the same shape
    """
    filtered_image = image.astype(np.float64)

    for _ in range(iterations):
        dx = np.gradient(filtered_image, axis=1)
        dy = np.gradient(filtered_image, axis=0)

        # The conduction slows down where the gradient is steep.
        c_x = 1 / (1 + (dx / kappa)**2)
        c_y = 1 / (1 + (dy / kappa)**2)

        filtered_image += delta_t * (
            c_x * np.roll(filtered_image, shift=-1, axis=1) +
            c_y * np.roll(filtered_image, shift=-1, axis=0) -
            (c_x + np.roll(c_x, shift=1, axis=1) + c_y + np.roll(c_y, shift=1, axis=0))
            * filtered_image
        )

    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def preprocess_image(image):
    """Clean a sonar image up for the two keypoint detectors.

    AKAZE reads the denoised and sharpened image at its own size. SuperPoint
    reads the same image resized to 480x640 and scaled into [0, 1], the size
    its frontend was trained at.

    The released eval scripts wrote four steps here, but only the last two
    reached the result: the min-max normalisation and the anisotropic diffusion
    were both overwritten, because every step read the original image again.
    This function keeps the two steps that did the work, so it reproduces the
    published numbers. :func:`anisotropic_diffusion` stays available for anyone
    who wants to put that step back.

    :param image: a uint8 greyscale sonar image
    :return: ``(norm_image, grayim)``, a uint8 image of the input shape and a
        float32 480x640 image in [0, 1]
    """
    norm_image = cv2.fastNlMeansDenoising(image, None, 5, 7, 21)
    norm_image = unsharp_mask(norm_image, kernel_size=(5, 5), sigma=0.5,
                              amount=2.0, threshold=0.0)
    norm_image = norm_image.astype(np.uint8)

    grayim = cv2.resize(norm_image, (640, 480), interpolation=cv2.INTER_AREA)
    grayim = grayim.astype(np.float32) / 255.

    return norm_image, grayim


def _keypoints_to_array(keypoints):
    """Return the (x, y) of each cv2.KeyPoint as an (N, 2) array."""
    if len(keypoints) == 0:
        # Keep the second axis, so the caller can index columns either way.
        return np.zeros((0, 2))
    return np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])


def generate_query_kpts(img, h, w, superpoint_weights=None):
    """Detect the query keypoints of one image with AKAZE and SuperPoint.

    The two detectors are complementary: AKAZE fires on the bright returns and
    SuperPoint on the texture between them. The network is asked about every
    point found, so nothing is subsampled here.

    :param img: a uint8 greyscale sonar image
    :param h: the image height the SuperPoint keypoints are rescaled to
    :param w: the image width the SuperPoint keypoints are rescaled to
    :param superpoint_weights: path to superpoint_v1.pth, or None for the copy
        in the repository
    :return: ``(coord, coord_akaze, coord_superpoint, desc_akaze,
        desc_superpoint)``. The three coordinate arrays are (N, 2) pixel
        positions and ``coord`` is the two of them stacked. ``desc_akaze`` is
        one row per AKAZE point, or None when there is none. ``desc_superpoint``
        is 256 x N, or None when SuperPoint found nothing.
    """
    superpoint = _get_superpoint(superpoint_weights)
    akaze = cv2.AKAZE_create()

    norm_image, grayim = preprocess_image(img)

    kpa, desc_akaze = akaze.detectAndCompute(norm_image, None)
    kps, desc_superpoint, _ = superpoint.run(grayim)
    # SuperPoint works at 480x640; put its points back on the sonar image.
    rescaled = rescale_keypoints(kps.T[:, :2], (h, w))

    coord_akaze = _keypoints_to_array(kpa)
    coord_superpoint = rescaled if len(rescaled) else np.zeros((0, 2))

    if len(coord_akaze) == 0:
        coord = coord_superpoint
    elif len(coord_superpoint) == 0:
        coord = coord_akaze
    else:
        coord = np.vstack((coord_akaze, coord_superpoint))

    return coord, coord_akaze, coord_superpoint, desc_akaze, desc_superpoint


def main():
    """Show the query points found on one sonar image."""
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--img', required=True, help='sonar image, .png or .npy')
    parser.add_argument('--meta', default=None, help='the matching _meta.npy file')
    parser.add_argument('--superpoint-weights', default=None,
                        help='path to superpoint_v1.pth')
    parser.add_argument('--out', default=None, help='save the figure here')
    args = parser.parse_args()

    meta = read_meta(args.meta) if args.meta else None
    img = load_image(args.img, meta)
    h, w = img.shape
    coord, coord_a, coord_s, _, _ = generate_query_kpts(
        img, h=h, w=w, superpoint_weights=args.superpoint_weights)
    print('{}: {} AKAZE + {} SuperPoint = {} query points'
          .format(os.path.basename(args.img), len(coord_a), len(coord_s), len(coord)))

    plt.imshow(img, cmap='gray')
    plt.scatter(coord_a[:, 0], coord_a[:, 1], s=2, c='r', label='AKAZE')
    plt.scatter(coord_s[:, 0], coord_s[:, 1], s=2, c='lime', label='SuperPoint')
    plt.legend(loc='upper right')
    plt.axis('off')
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(args.out, bbox_inches='tight', dpi=150)
        print('saved {}'.format(args.out))
    else:
        plt.show()


if __name__ == '__main__':
    main()
