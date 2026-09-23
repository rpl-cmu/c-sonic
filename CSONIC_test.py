"""Match a sonar image pair with a released C-SONIC checkpoint.

This is the downstream entry point. Give it two sonar images, their metadata
and a checkpoint, and it returns the pixel correspondences the network predicts.
With the two poses it also reports how far those matches sit from where the
geometry says they should be.

    uv run python CSONIC_test.py \
        --img1 H0/3.png --meta1 H0/3_meta.npy \
        --img2 L0/4.png --meta2 L0/4_meta.npy \
        --pose1 H0/3.npy --pose2 L0/4.npy \
        --model pretrained/CSONIC_pretrained.pth --ransac --out match.png

There are three inference modes. Both defaults are deterministic:

* ``--bn batch`` (the default) normalises the input as training did and lets
  batch normalisation use the statistics of the image in front of it. The two
  halves of a cross-modal pair sit at different intensity scales, and this is
  what absorbs the difference.
* ``--bn running`` is plain eval mode, using the averages stored in the
  checkpoint. It matches worse on cross-modal pairs; see results.md.
* ``--legacy-input`` reproduces the scripts that came with the paper: raw pixel
  values and the whole network in train mode, dropout included. Use it to
  reproduce published numbers, not for repeatable work.

Two filters choose the query points before the network sees them. The
brightness filter, on by default, drops the points in the dark water column,
where there is no return to match (:func:`filter_dark_queries`; off with
``--legacy-input``). ``--visible-only`` uses the two poses to drop the points
that sonar 2 cannot see (:func:`visible_in_image2`).

The geometry helpers keep the names they had in the evaluation code that came
with the paper, so an old script is easy to port.

Two conventions run through this file:

* A *pose* is the 4x4 sensor-to-world matrix stored next to each image. An
  *extrinsic*, written ``T``, is its inverse, which is what the loss and the
  epipolar cost work in. :func:`pose_correction` turns one into the other.
* Pixels are ``(u, v)``, polar coordinates are ``(bearing, range)`` with the
  bearing in radians and the range in metres.
"""
import argparse
import contextlib
import os
import time

import cv2
import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

import config
import detection
from CSONIC.csonic_model import CSONICModel
# The earlier name of zero_subnormal_weights. Code that calls it still works.
# CSONICModel.load_model now zeroes the weights of every checkpoint it loads.
from CSONIC.csonic_model import zero_subnormal_weights as _zero_subnormal_weights  # noqa: F401
from dataloader.sonardata import SONAR_MEAN, SONAR_STD
from utils import make_matching_figure

# The number of elevation samples the released model was trained with. The
# epipolar cost takes the closest point on that arc.
DEFAULT_NUM_SAMPLES = 100

# How batch normalisation gets its statistics during inference.
#   'running' uses the averages stored in the checkpoint, the usual eval mode.
#   'batch'   uses the statistics of the image in front of it.
# The two halves of a cross-modal pair sit at different intensity scales, and
# 'batch' absorbs that difference where the stored averages cannot. Measured on
# the real-world pairs it matches better, so it is the default. Dropout stays
# off in both, so both are deterministic. See results.md for the numbers.
BN_MODES = ('running', 'batch')
DEFAULT_BN_MODE = 'batch'

# The brightness filter of the query points. A point is kept when the mean of
# image 1 (uint8, the image the network gets) over a 15 x 15 box around it is
# at least 15. Below that the point sits in the dark water column, where there
# is no return to match. Calibrated on the 25 released sample pairs in both
# directions (10889 query points): the rule drops 253 of them (2.3 %). Of the
# dropped points with a true match in image 2, 48 % land within 20 px of it,
# against 76 % of the points the rule keeps. A rule relative to the
# mean of the bottom rows, like the shadow mask of the training loss, does not
# work here: that mean ranges from 0 to 55 across the images.
DEFAULT_MIN_INTENSITY = 15
DEFAULT_INTENSITY_WINDOW = 15

# How far outside the view of sonar 2 a point may sit and still count as
# inside, in radians and in metres. The edge pixels of an image lie exactly on
# the edge of its view, and rounding must not push them out.
_VIEW_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# model and images
# --------------------------------------------------------------------------

def _default_device():
    """The device CSONICModel picks for itself."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path, device=None, cross_attention=1,
               zero_subnormal_weights=True):
    """Load a checkpoint and return the model, ready for inference.

    :param model_path: the ``.pth`` file to load
    :param device: build the model on this device, for example ``'cpu'`` or
        ``'cuda:1'``. The default is the GPU when there is one.
    :param cross_attention: 1 for C-SONIC, 0 for the SONIC baseline. It has to
        match the checkpoint, or the weights will not load.
    :param zero_subnormal_weights: zero every weight smaller than the smallest
        normal float, which is what keeps CPU inference fast. It is passed to
        :class:`CSONICModel`, which does the zeroing when it loads the
        checkpoint; see :func:`CSONIC.csonic_model.zero_subnormal_weights`.
        False keeps the stored values. Nothing process-wide is changed.
    :return: a :class:`CSONICModel` in eval mode

    Loading changes nothing outside the model it returns: the device is chosen
    before anything is built, so no memory is allocated on any other device,
    and the random number generators are restored on the way out.
    """
    target = str(device) if device is not None else _default_device()

    args = config.get_args([])
    args.ckpt_path = str(model_path)
    args.pretrained = 0
    args.phase = 'test'
    args.cross_attention = cross_attention

    # Building the network draws from the global generators to initialise the
    # weights, which the checkpoint then overwrites. Fork them so a caller's
    # own sequence of random numbers is not shifted by loading a model.
    # Fork only the generator of the device the model is built on. A CPU load
    # must not touch CUDA at all, not even to ask which GPU is current.
    target_device = torch.device(target)
    if target_device.type == 'cuda':
        index = target_device.index
        forked = [index if index is not None else torch.cuda.current_device()]
    else:
        forked = []
    with torch.random.fork_rng(devices=forked):
        model = CSONICModel(args, device=target,
                            zero_subnormal_weights=zero_subnormal_weights)

    model.model.eval()
    return model


def prepare_image(img_uint8, device, normalize=True):
    """Put one sonar image in the layout the network takes.

    :param img_uint8: a uint8 greyscale image
    :param device: where to put the tensor
    :param normalize: True scales to [0, 1] and applies the training mean and
        standard deviation, which is what the network was trained on. False
        passes the stored pixel values straight through, which is what the
        paper-era scripts did.
    :return: a float tensor of shape (1, 1, H, W)
    """
    image = torch.from_numpy(np.ascontiguousarray(img_uint8)).float()
    if normalize:
        image = (image / 255.0 - SONAR_MEAN) / SONAR_STD
    return image.unsqueeze(0).unsqueeze(0).to(device)


def filter_dark_queries(coord, img, min_intensity=DEFAULT_MIN_INTENSITY,
                        window=DEFAULT_INTENSITY_WINDOW):
    """Tell which query points sit on a return, and which in the dark.

    A point is kept when the mean of ``img`` over a ``window`` x ``window`` box
    centred on its nearest pixel is at least ``min_intensity``; the threshold
    itself counts as bright. The box takes the border handling of
    ``cv2.boxFilter``, which reflects the image at its edges. A point on the
    border, or outside the image, reads the nearest pixel inside it.

    :param coord: pixel positions ``(u, v)``, (N, 2)
    :param img: the uint8 image the points were detected in, the same image
        the network gets
    :param min_intensity: the smallest box mean that is kept, on the 0-255
        scale. 0 or less keeps every point.
    :param window: the side of the box, in pixels; a positive odd number
    :return: a boolean mask of length N, True for the points to keep
    """
    if (isinstance(window, bool) or not isinstance(window, (int, np.integer))
            or window < 1 or window % 2 == 0):
        raise ValueError('window must be a positive odd number of pixels, got {!r}'
                         .format(window))

    if not np.isfinite(min_intensity):
        raise ValueError('min_intensity must be a finite number, got {!r}'
                         .format(min_intensity))

    coord = np.asarray(coord, dtype=float).reshape(-1, 2)
    if min_intensity <= 0 or len(coord) == 0:
        return np.ones(len(coord), dtype=bool)

    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError('img must be a greyscale image, got shape {}'.format(img.shape))
    h, w = img.shape

    # Compare the sum of the box with the threshold times its area. The sum of
    # uint8 values is a whole number, exact in float64, so a box whose mean is
    # exactly the threshold is kept whatever the rounding of a division.
    box_sum = cv2.boxFilter(img, cv2.CV_64F, (window, window), normalize=False)
    u = np.clip(np.rint(coord[:, 0]), 0, w - 1).astype(int)
    v = np.clip(np.rint(coord[:, 1]), 0, h - 1).astype(int)
    return box_sum[v, u] >= float(min_intensity) * window * window


def query_points(im1, meta1=None, kpt_roi=None, superpoint_weights=None,
                 min_intensity=None, window=DEFAULT_INTENSITY_WINDOW,
                 visibility=None, return_counts=False):
    """Detect the points in image 1 that the network will be asked about.

    Three filters run after the detection, in this order: the ROI box, the
    brightness filter (:func:`filter_dark_queries`) and the view filter
    (:func:`visible_in_image2`).

    :param im1: the uint8 image
    :param meta1: its metadata, used for the size the SuperPoint points are
        rescaled to. The image shape is used when it is None.
    :param kpt_roi: keep only the points inside ``(u_min, u_max, v_min, v_max)``.
        The real-world scenes only have returns in a band of the image, so this
        is how the released evaluation cut the rest away.
    :param superpoint_weights: path to superpoint_v1.pth, or None for the copy
        in the repository
    :param min_intensity: drop the points whose box mean in ``im1`` is below
        this. None means :data:`DEFAULT_MIN_INTENSITY`; 0 turns the filter off.
    :param window: the side of that box, in pixels
    :param visibility: None, or ``(pose1, pose2, meta2)``: the 4x4
        sensor-to-world poses of the two images and the metadata of image 2.
        With it, the points that sonar 2 cannot see are dropped. It needs
        ``meta1``. It uses the true poses, so it fits an evaluation, or a run
        with a pose prior from odometry, not a blind match.
    :param return_counts: also return how many points each step dropped
    :return: an (N, 2) float array of pixel positions. With ``return_counts``,
        ``(coord, counts)``, where ``counts`` holds ``detected`` and ``kept``,
        the points before and after the filters, and ``roi``, ``dark`` and
        ``not_visible``, the points each filter dropped.
    """
    if visibility is not None:
        if meta1 is None:
            raise ValueError('visibility needs meta1, the metadata of image 1')
        pose1, pose2, meta2 = visibility
        if pose1 is None or pose2 is None or meta2 is None:
            raise ValueError('visibility needs (pose1, pose2, meta2), got a None in it')
    if min_intensity is None:
        min_intensity = DEFAULT_MIN_INTENSITY

    if meta1 is not None:
        h, w = int(meta1['height']), int(meta1['width'])
    else:
        h, w = im1.shape[:2]

    coord = detection.generate_query_kpts(
        im1, h=h, w=w, superpoint_weights=superpoint_weights)[0]
    coord = np.asarray(coord, dtype=float).reshape(-1, 2)
    counts = {'detected': len(coord), 'roi': 0, 'dark': 0, 'not_visible': 0}

    if kpt_roi is not None:
        u_min, u_max, v_min, v_max = kpt_roi
        inside = ((coord[:, 0] >= u_min) & (coord[:, 0] <= u_max)
                  & (coord[:, 1] >= v_min) & (coord[:, 1] <= v_max))
        counts['roi'] = int(np.sum(~inside))
        coord = coord[inside]

    bright = filter_dark_queries(coord, im1, min_intensity=min_intensity, window=window)
    counts['dark'] = int(np.sum(~bright))
    coord = coord[bright]

    if visibility is not None:
        visible = visible_in_image2(coord, pose1, pose2, meta1, meta2)
        counts['not_visible'] = int(np.sum(~visible))
        coord = coord[visible]

    counts['kept'] = len(coord)
    if return_counts:
        return coord, counts
    return coord


def _resolve_min_intensity(min_intensity, normalize_input):
    """Return the brightness threshold a run uses.

    An explicit value always wins. None means :data:`DEFAULT_MIN_INTENSITY`,
    except on the legacy path, where it means 0: the thesis pipeline had no
    brightness filter, and the legacy path must still reproduce it.
    """
    if min_intensity is not None:
        return min_intensity
    return DEFAULT_MIN_INTENSITY if normalize_input else 0


@contextlib.contextmanager
def _frozen_batchnorm_stats(net):
    """Stop batch normalisation from updating its running statistics.

    In train mode batch normalisation normalises by the statistics of the batch
    and folds them into its running averages, even under ``no_grad``. Inference
    must not leave that mark on the model: a later eval-mode call would use the
    disturbed averages and return different matches. Turning the tracking off
    keeps the batch statistics, which is what the paper-era scripts computed,
    and leaves the stored averages exactly as the checkpoint had them.
    """
    layers = [m for m in net.modules() if isinstance(m, _BatchNorm)]
    saved = [m.track_running_stats for m in layers]
    for layer in layers:
        layer.track_running_stats = False
    try:
        yield
    finally:
        for layer, flag in zip(layers, saved):
            layer.track_running_stats = flag


@contextlib.contextmanager
def _batch_statistics(net):
    """Let batch normalisation use the statistics of the image in front of it.

    Only the batch normalisation layers go into train mode. Dropout stays in
    eval, so the forward pass is still deterministic, and the running averages
    are left as the checkpoint had them.
    """
    layers = [m for m in net.modules() if isinstance(m, _BatchNorm)]
    saved = [(m.training, m.track_running_stats) for m in layers]
    for layer in layers:
        layer.train(True)
        layer.track_running_stats = False
    try:
        yield
    finally:
        for layer, (training, track) in zip(layers, saved):
            layer.train(training)
            layer.track_running_stats = track


def match_points(model, im1, im2, coord1, th=0.1, normalize_input=True,
                 bn_mode=DEFAULT_BN_MODE):
    """Ask the network where each query point went, and keep the sure ones.

    The matches come back sorted by ``std_f``, the spread of the fine-level
    correspondence distribution, smallest first. A small spread means the
    network put its probability mass in one place.

    :param model: a loaded :class:`CSONICModel`
    :param im1: the first uint8 image
    :param im2: the second uint8 image
    :param coord1: the query points in image 1, (N, 2)
    :param th: keep the matches whose ``std_f`` is below this. The value is in
        normalised image units, so 0.1 is about 26 px on a 512 px image.
    :param normalize_input: True for the training-time input pipeline. False
        reproduces the paper-era scripts exactly: the raw pixel values, and the
        whole network in train mode, so batch normalisation runs on the batch
        and ``Dropout2d(0.4)`` stays on. Dropout makes that path vary from run
        to run, and it ignores ``bn_mode``.
    :param bn_mode: where batch normalisation takes its statistics from, one of
        :data:`BN_MODES`. ``'running'`` is the stored averages, ``'batch'`` the
        statistics of the image in front of it. Both keep dropout off, so both
        are deterministic.
    :return: ``(mkpts0, mkpts1, std_f)`` as numpy arrays of the same length

    Whichever path runs, the model comes back in eval mode with its running
    statistics untouched, so the paths can be mixed freely.
    """
    if bn_mode not in BN_MODES:
        raise ValueError('bn_mode must be one of {}, got {!r}'
                         .format(BN_MODES, bn_mode))

    coord1 = np.asarray(coord1, dtype=float).reshape(-1, 2)
    if len(coord1) == 0:
        empty = np.zeros((0, 2))
        return empty, empty.copy(), np.zeros(0)

    tensor1 = prepare_image(im1, model.device, normalize=normalize_input)
    tensor2 = prepare_image(im2, model.device, normalize=normalize_input)
    kpts = torch.from_numpy(coord1).float().unsqueeze(0).to(model.device)

    if not normalize_input:
        # The old scripts never called eval(), so reproducing them means
        # putting the whole network back in train mode, dropout included.
        mode = _frozen_batchnorm_stats(model.model)
    elif bn_mode == 'batch':
        mode = _batch_statistics(model.model)
    else:
        mode = contextlib.nullcontext()

    try:
        model.model.train(not normalize_input)
        with mode, torch.no_grad():
            out = model.model(tensor1, tensor2, kpts)
    finally:
        model.model.eval()

    std_f = out['std_f'].squeeze(0).detach().cpu().numpy()
    mkpts1 = out['coord2_ef'].squeeze(0).detach().cpu().numpy()

    order = np.argsort(std_f)
    std_f = std_f[order]
    mkpts1 = mkpts1[order]
    mkpts0 = coord1[order]

    keep = std_f < th
    return mkpts0[keep], mkpts1[keep], std_f[keep]


def expectation_matching(img1_path, img2_path, meta1_path, meta2_path, model=None,
                         model_path=None, th=0.1, kpt_roi=None, normalize_input=True,
                         superpoint_weights=None, bn_mode=DEFAULT_BN_MODE,
                         min_intensity=None, poses=None):
    """Match one sonar pair, from file paths to pixel correspondences.

    :param img1_path: the first sonar image, ``.png`` or ``.npy``
    :param img2_path: the second sonar image
    :param meta1_path: the ``_meta.npy`` of the first image
    :param meta2_path: the ``_meta.npy`` of the second image
    :param model: an already loaded model, to match many pairs with one load
    :param model_path: a checkpoint to load when ``model`` is None
    :param th: the ``std_f`` threshold; see :func:`match_points`
    :param kpt_roi: restrict the query points to ``(u_min, u_max, v_min, v_max)``
    :param normalize_input: see :func:`match_points`
    :param superpoint_weights: path to superpoint_v1.pth
    :param bn_mode: see :func:`match_points`
    :param min_intensity: the brightness threshold of the query points; see
        :func:`query_points`. None means :data:`DEFAULT_MIN_INTENSITY`, or 0
        when ``normalize_input`` is False, so the legacy path still reproduces
        the thesis pipeline.
    :param poses: ``(pose1, pose2)``, the 4x4 sensor-to-world poses of the two
        images, to drop the query points that sonar 2 cannot see; see
        :func:`visible_in_image2`. The metadata comes from ``meta1_path`` and
        ``meta2_path``. None keeps them.
    :return: ``(mkpts0, mkpts1, std_f)``; the pixels in image 1, the matching
        pixels in image 2, and the spread of each match
    """
    if poses is not None:
        pose1, pose2 = poses
        if pose1 is None or pose2 is None:
            raise ValueError('poses must be (pose1, pose2), got a None in it')
    if model is None:
        if model_path is None:
            raise ValueError('give either a loaded model or a model_path')
        model = load_model(model_path)

    data = detection.load_data(img1_path, img2_path, meta1_path=meta1_path,
                               meta2_path=meta2_path)
    visibility = (pose1, pose2, data['meta2']) if poses is not None else None
    coord1 = query_points(data['im1'], data['meta1'], kpt_roi=kpt_roi,
                          superpoint_weights=superpoint_weights,
                          min_intensity=_resolve_min_intensity(min_intensity,
                                                               normalize_input),
                          visibility=visibility)
    return match_points(model, data['im1'], data['im2'], coord1, th=th,
                        normalize_input=normalize_input, bn_mode=bn_mode)


# --------------------------------------------------------------------------
# sonar geometry
# --------------------------------------------------------------------------

def pix_to_polar_batch(point, range_max=10.0, range_min=0.1, image_width=512.0,
                       image_height=512.0, bearing_max=130.0):
    """Convert pixels to polar coordinates.

    :param point: pixel coordinates (u, v), (N, 2)
    :param range_max: far range of the image, in metres
    :param range_min: near range of the image, in metres
    :param image_width: image width, in pixels
    :param image_height: image height, in pixels
    :param bearing_max: horizontal field of view (azimuth), in degrees
    :return: the bearing in radians and the range in metres, each (N,)
    """
    point = np.asarray(point, dtype=float)
    u = point[:, 0]
    v = point[:, 1]
    bearing = ((image_width - u) * np.deg2rad(bearing_max) / image_width) \
        - np.deg2rad(bearing_max) / 2
    ranges = ((range_max - range_min) / image_height) * (image_height - v) + range_min
    return bearing, ranges


def polar_to_pix_batch(point, image_width=512.0, image_height=512.0, bearing_max=130.0,
                       range_max=10.0, range_min=0.1):
    """Convert polar coordinates back to pixels.

    :param point: a ``(bearing, range)`` pair of arrays of the same shape, the
        bearing in radians and the range in metres
    :return: the pixel coordinates u and v, shaped like the inputs
    """
    bearing = np.asarray(point[0], dtype=float)
    ranges = np.asarray(point[1], dtype=float)
    v = image_height - ((ranges - range_min) * image_height) / (range_max - range_min)
    u = image_width - (bearing + np.deg2rad(bearing_max) / 2) * image_width \
        / np.deg2rad(bearing_max)
    return u, v


def polar_to_cartesian_sampled_points(ranges, bearings, phi, sample_points=1):
    """Sample the arc of elevation ambiguity of each point, in 3D.

    A sonar measures range and bearing but not elevation, so one pixel is a
    whole arc in space. This walks that arc.

    :param ranges: range of each point, in metres, (N,) or a scalar
    :param bearings: bearing of each point, in radians, same shape as ``ranges``
    :param phi: the vertical field of view, in degrees. The arc is sampled over
        ``[-phi/2, +phi/2]``; pass 0 for the centre of the arc alone.
    :param sample_points: how many samples to take along the arc
    :return: cartesian points in metres, shape ``(3, ..., sample_points)``,
        with x, y and z along the first axis. This matches the convention of
        ``CtoFCriterion.convert_to_arc``, which trained the model.
    """
    sampled_phi = np.deg2rad(np.linspace(-phi / 2.0, phi / 2.0, sample_points))
    r = np.asarray(ranges, dtype=float)[..., None]
    b = np.asarray(bearings, dtype=float)[..., None]

    x = r * np.cos(b) * np.cos(sampled_phi)
    y = r * np.sin(b) * np.cos(sampled_phi)
    z = -r * np.sin(sampled_phi) * np.ones_like(b)
    return np.array([x, y, z])


def cart_to_polar_batch(points):
    """Return the bearing (radians) and the range (metres) of cartesian points.

    :param points: shape ``(3, ...)``, with x, y and z along the first axis
    """
    points = np.asarray(points, dtype=float)
    ranges = np.linalg.norm(points, axis=0)
    bearings = np.arctan2(points[1], points[0])
    return bearings, ranges


def pose_correction(pose):
    """Turn a sensor-to-world pose into the extrinsic matrix.

    The rotation is transposed and the translation is rotated back, so the
    result maps world points into the sonar frame. This is what
    ``SonarData.get_extrinsics`` does during training.

    :param pose: a 4x4 pose matrix
    :return: the 4x4 extrinsic matrix
    """
    pose = np.asarray(pose, dtype=float)
    corrected = np.eye(4)
    rotation = pose[:3, :3].T
    corrected[:3, :3] = rotation
    corrected[:3, 3] = -rotation @ pose[:3, 3]
    return corrected


def transform_points(points, pose0, pose1):
    """Move cartesian points from the first sonar frame into the second.

    :param points: shape ``(3, ...)``, in metres, x/y/z along the first axis
    :param pose0: the extrinsic matrix of the first image
    :param pose1: the extrinsic matrix of the second image
    :return: the moved points, shaped like the input
    """
    points = np.asarray(points, dtype=float)
    rel_pose = np.asarray(pose1, dtype=float) @ np.linalg.inv(np.asarray(pose0, dtype=float))
    rotation = rel_pose[:3, :3]
    translation = rel_pose[:3, 3]
    moved = np.einsum('ij,j...->i...', rotation, points)
    return moved + translation.reshape((3,) + (1,) * (points.ndim - 1))


def project_keypoints(mkpts0, pose1, pose2, meta1, meta2):
    """Project the query pixels of image 1 into image 2 using the two poses.

    The elevation of a sonar return is unknown, so this takes the centre of the
    arc, elevation zero. The result is where each point would land if it lay in
    the plane of the sensor: the reference the predicted matches are scored
    against.

    :param mkpts0: pixels in image 1, (N, 2)
    :param pose1: the 4x4 sensor-to-world pose of image 1
    :param pose2: the 4x4 sensor-to-world pose of image 2
    :param meta1: the metadata of image 1
    :param meta2: the metadata of image 2
    :return: the projected pixels in image 2, (N, 2)
    """
    bearings, ranges = pix_to_polar_batch(mkpts0,
                                          range_max=meta1['r_max'],
                                          range_min=meta1['r_min'],
                                          image_width=meta1['width'],
                                          image_height=meta1['height'],
                                          bearing_max=meta1['azi'])

    points = polar_to_cartesian_sampled_points(ranges, bearings, 0.0, 1)
    moved = transform_points(points, pose_correction(pose1), pose_correction(pose2))
    new_bearings, new_ranges = cart_to_polar_batch(moved)

    u, v = polar_to_pix_batch((new_bearings, new_ranges),
                              image_width=meta2['width'],
                              image_height=meta2['height'],
                              bearing_max=meta2['azi'],
                              range_max=meta2['r_max'],
                              range_min=meta2['r_min'])
    return np.stack([np.ravel(u), np.ravel(v)], axis=1)


def visible_in_image2(coord, pose1, pose2, meta1, meta2,
                      num_samples=DEFAULT_NUM_SAMPLES):
    """Tell which pixels of image 1 sonar 2 can see, using the two poses.

    A sonar pixel is an arc of elevation, not a point, so the arc of each query
    is sampled over ``[-elev1/2, +elev1/2]`` and moved into the frame of sonar
    2, the same way as in :func:`project_keypoints` and
    :func:`sonar_epipolar_cost_np`. The point is kept when at least one sample
    lies in the view of sonar 2: a bearing within ``+-azi2/2``, a range from
    ``r_min2`` to ``r_max2``, and an elevation within ``+-elev2/2``. The
    elevation is the angle of the sample out of the x-y plane of sonar 2, and
    only its size counts, so the sign convention of z does not matter. A point
    exactly on the edge of the view counts as inside. The arc is sampled, not
    solved: when only a sliver of it, narrower than the step between two
    samples, lies in the view, the point can be dropped although sonar 2 sees
    it. With the default 100 samples that step is about 0.2 degrees of
    elevation for a 20-degree arc.

    This uses the true poses. It fits an evaluation, or a run with a pose prior
    from odometry; a blind match has no poses to give it.

    :param coord: pixels in image 1, (N, 2)
    :param pose1: the 4x4 sensor-to-world pose of image 1
    :param pose2: the 4x4 sensor-to-world pose of image 2
    :param meta1: the metadata of image 1
    :param meta2: the metadata of image 2
    :param num_samples: samples along each arc, at least 2 so that both of its
        ends are in
    :return: a boolean mask of length N, True for the points sonar 2 can see
    """
    if num_samples < 2:
        raise ValueError('num_samples must be at least 2, got {}'.format(num_samples))
    coord = np.asarray(coord, dtype=float).reshape(-1, 2)
    if len(coord) == 0:
        return np.zeros(0, dtype=bool)

    bearings, ranges = pix_to_polar_batch(coord,
                                          range_max=meta1['r_max'],
                                          range_min=meta1['r_min'],
                                          image_width=meta1['width'],
                                          image_height=meta1['height'],
                                          bearing_max=meta1['azi'])
    arc = polar_to_cartesian_sampled_points(ranges, bearings, float(meta1['elev']),
                                            num_samples)
    moved = transform_points(arc, pose_correction(pose1), pose_correction(pose2))
    bearing2, range2 = cart_to_polar_batch(moved)  # (N, num_samples)
    elevation2 = np.arctan2(moved[2], np.hypot(moved[0], moved[1]))

    tol = _VIEW_TOLERANCE
    in_view = ((np.abs(bearing2) <= np.deg2rad(meta2['azi']) / 2 + tol)
               & (range2 >= meta2['r_min'] - tol)
               & (range2 <= meta2['r_max'] + tol)
               & (np.abs(elevation2) <= np.deg2rad(meta2['elev']) / 2 + tol))
    return in_view.any(axis=1)


def sonar_epipolar_cost_np(mkpts0, mkpts1, meta1, meta2, T1, T2,
                           num_samples=DEFAULT_NUM_SAMPLES):
    """Return how far each match sits from its epipolar arc, squared, in m^2.

    This is the numpy form of the epipolar term the model was trained with. The
    arc of elevation ambiguity of the query point is sampled, moved into the
    second sonar frame, and the closest sample to the predicted match is taken.

    :param mkpts0: query pixels in image 1, (N, 2)
    :param mkpts1: the matching pixels in image 2, (N, 2)
    :param meta1: the metadata of image 1
    :param meta2: the metadata of image 2
    :param T1: the 4x4 *extrinsic* of image 1, as :func:`pose_correction` returns
    :param T2: the 4x4 extrinsic of image 2
    :param num_samples: samples along the elevation arc
    :return: the squared distance of each match, (N,)
    """
    bearing1, range1 = pix_to_polar_batch(mkpts0,
                                          range_max=meta1['r_max'],
                                          range_min=meta1['r_min'],
                                          image_width=meta1['width'],
                                          image_height=meta1['height'],
                                          bearing_max=meta1['azi'])
    bearing2, range2 = pix_to_polar_batch(mkpts1,
                                          range_max=meta2['r_max'],
                                          range_min=meta2['r_min'],
                                          image_width=meta2['width'],
                                          image_height=meta2['height'],
                                          bearing_max=meta2['azi'])

    arc = polar_to_cartesian_sampled_points(range1, bearing1, meta1['elev'], num_samples)
    moved = transform_points(arc, T1, T2)
    bearings_t, ranges_t = cart_to_polar_batch(moved)  # (N, num_samples)

    # The law of cosines, in the plane the sonar images: the distance between
    # the arc sample and the match, both as (bearing, range) pairs.
    dist_squared = ranges_t**2 + range2[:, None]**2 \
        - 2 * ranges_t * range2[:, None] * np.cos(bearing2[:, None] - bearings_t)
    return np.min(dist_squared, axis=1)


def ransac_outlier_rejection(mkpts0, mkpts1, pose1, pose2, meta1, meta2,
                             num_iterations=500, threshold=0.1, rng=None):
    """Return the indices of the matches that agree with the two poses.

    The poses are known, so there is no model to fit here. What the random
    subsets decide is the scale the epipolar cost is measured against: each
    iteration draws a quarter of the matches, takes the largest cost among
    them, and keeps every match whose cost is below ``threshold`` times that
    scale. The largest set of the run wins, so with enough iterations the scale
    settles on the largest cost in the whole set and ``threshold`` is what
    controls how strict the rejection is.

    :param mkpts0: query pixels in image 1, (N, 2)
    :param mkpts1: the matching pixels in image 2, (N, 2)
    :param pose1: the 4x4 sensor-to-world pose of image 1
    :param pose2: the 4x4 sensor-to-world pose of image 2
    :param meta1: the metadata of image 1
    :param meta2: the metadata of image 2
    :param num_iterations: how many subsets to try
    :param threshold: the fraction of the scale a match may cost
    :param rng: a ``numpy.random.Generator``, for a reproducible result
    :return: an int array of indices into ``mkpts0``
    """
    mkpts0 = np.asarray(mkpts0, dtype=float)
    total = len(mkpts0)
    if total == 0:
        return np.zeros(0, dtype=int)

    rng = np.random.default_rng() if rng is None else rng

    # The cost of a match depends only on that match, so it is the same in
    # every subset. Computing it once gives the same answer much faster.
    cost = sonar_epipolar_cost_np(mkpts0, mkpts1, meta1, meta2,
                                  pose_correction(pose1), pose_correction(pose2))
    if not np.isfinite(cost).any():
        # No match has a cost that can be compared with anything. Keeping them
        # all would be a claim the data does not support, so keep none.
        return np.zeros(0, dtype=int)
    if np.nanmax(cost) <= 0:
        # Every match is exactly on its arc; there is nothing to reject.
        return np.arange(total)

    best = np.zeros(0, dtype=int)
    subset_size = max(1, total // 4)
    for _ in range(num_iterations):
        subset = rng.choice(total, size=subset_size, replace=True)
        scale = np.nanmax(cost[subset])
        if not np.isfinite(scale) or scale <= 0:
            continue
        inliers = np.where(cost < threshold * scale)[0]
        if len(inliers) > len(best):
            best = inliers
    return best


def inlier_percentage(gt_pix, pred_pix, px_th=20):
    """Compare predicted matches with their projected positions, row by row.

    Row i of ``pred_pix`` is scored against row i of ``gt_pix``, so pass the
    output of :func:`project_keypoints` for the same query points.

    :param gt_pix: the projected positions, (N, 2)
    :param pred_pix: the predicted matches, (N, 2)
    :param px_th: a match counts as an inlier below this many pixels
    :return: ``(inlier_idx, percentage, mean_dist, std_dist)``; the indices of
        the inliers, their share of all matches in percent, and the mean and
        the standard deviation of the pixel distance over all matches
    """
    gt_pix = np.asarray(gt_pix, dtype=float)
    pred_pix = np.asarray(pred_pix, dtype=float)
    if gt_pix.shape != pred_pix.shape:
        raise ValueError('gt_pix and pred_pix must line up, got {} and {}'
                         .format(gt_pix.shape, pred_pix.shape))
    if len(gt_pix) == 0:
        return np.zeros(0, dtype=int), 0.0, float('nan'), float('nan')

    distances = np.linalg.norm(pred_pix - gt_pix, axis=1)
    inlier_idx = np.where(distances < px_th)[0]
    percentage = 100.0 * len(inlier_idx) / len(distances)
    return inlier_idx, percentage, float(distances.mean()), float(distances.std())


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Match one sonar pair with a C-SONIC checkpoint.')
    parser.add_argument('--img1', required=True, help='first sonar image')
    parser.add_argument('--img2', required=True, help='second sonar image')
    parser.add_argument('--meta1', required=True, help='metadata of the first image')
    parser.add_argument('--meta2', required=True, help='metadata of the second image')
    parser.add_argument('--pose1', default=None,
                        help='4x4 pose of the first image; enables the error report')
    parser.add_argument('--pose2', default=None, help='4x4 pose of the second image')
    parser.add_argument('--model', required=True, help='the checkpoint to load')
    parser.add_argument('--out', default='match.png', help='where to save the figure')
    parser.add_argument('--th', type=float, default=1.0,
                        help='keep the matches with std_f below this')
    parser.add_argument('--ransac', action='store_true',
                        help='reject the matches that disagree with the poses')
    parser.add_argument('--ransac-iters', type=int, default=500,
                        help='RANSAC iterations')
    parser.add_argument('--ransac-th', type=float, default=0.1,
                        help='RANSAC inlier threshold, as a fraction of the cost scale')
    parser.add_argument('--roi', type=float, nargs=4, default=None,
                        metavar=('U0', 'U1', 'V0', 'V1'),
                        help='keep only the query points inside this box')
    parser.add_argument('--min-intensity', type=float, default=None,
                        help='drop the query points where the mean of image 1 over '
                             'the --intensity-window box is below this value (0-255). '
                             'These points sit in the dark water column, where there '
                             'is no return to match. Default: {}, or 0 with '
                             '--legacy-input. 0 turns the filter off'
                             .format(DEFAULT_MIN_INTENSITY))
    parser.add_argument('--intensity-window', type=int,
                        default=DEFAULT_INTENSITY_WINDOW,
                        help='the side of the box of --min-intensity, in pixels; an '
                             'odd number (default: %(default)s)')
    parser.add_argument('--visible-only', action='store_true',
                        help='use the two poses to drop the query points that sonar 2 '
                             'cannot see. Needs --pose1 and --pose2. For an evaluation '
                             'or a run with a pose prior, not for a blind match')
    parser.add_argument('--px-th', type=float, default=20.0,
                        help='pixel distance below which a match counts as an inlier')
    parser.add_argument('--bn', choices=BN_MODES, default=DEFAULT_BN_MODE,
                        help='where batch normalisation takes its statistics from: '
                             '"batch" from the image in front of it, "running" from '
                             'the averages stored in the checkpoint. Both keep '
                             'dropout off, so both are deterministic. Ignored with '
                             '--legacy-input (default: %(default)s)')
    parser.add_argument('--legacy-input', action='store_true',
                        help='reproduce the paper-era pipeline: raw pixel values and '
                             'the whole network in train mode, dropout included. '
                             'Not deterministic. It also turns the brightness filter '
                             'off, unless --min-intensity is given')
    parser.add_argument('--superpoint-weights', default=None,
                        help='path to superpoint_v1.pth')
    parser.add_argument('--cross-attention', type=int, default=1,
                        help='1 for C-SONIC, 0 for the SONIC baseline')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed of the RANSAC random generator')
    return parser.parse_args(argv)


def _describe_counts(counts, roi, dark, visible):
    """Say how many query points each active filter dropped, in one line."""
    parts = ['{} detected'.format(counts['detected'])]
    if roi:
        parts.append('{} outside --roi'.format(counts['roi']))
    if dark:
        parts.append('{} dark'.format(counts['dark']))
    if visible:
        parts.append('{} not visible in image 2'.format(counts['not_visible']))
    parts.append('{} kept'.format(counts['kept']))
    return ', '.join(parts)


def main(argv=None):
    args = parse_args(argv)
    if args.visible_only and (args.pose1 is None or args.pose2 is None):
        raise ValueError('--visible-only needs --pose1 and --pose2')

    data = detection.load_data(args.img1, args.img2, args.pose1, args.pose2,
                               args.meta1, args.meta2)
    model = load_model(args.model, cross_attention=args.cross_attention)

    started = time.time()
    min_intensity = _resolve_min_intensity(args.min_intensity, not args.legacy_input)
    visibility = ((data['pose1'], data['pose2'], data['meta2'])
                  if args.visible_only else None)
    coord1, counts = query_points(data['im1'], data['meta1'], kpt_roi=args.roi,
                                  superpoint_weights=args.superpoint_weights,
                                  min_intensity=min_intensity,
                                  window=args.intensity_window,
                                  visibility=visibility, return_counts=True)
    mkpts0, mkpts1, std_f = match_points(model, data['im1'], data['im2'], coord1,
                                         th=args.th,
                                         normalize_input=not args.legacy_input,
                                         bn_mode=args.bn)
    print('mode: {}'.format('legacy (raw input, train mode, dropout on)'
                            if args.legacy_input else
                            'normalised input, batch norm from {} statistics'
                            .format(args.bn)))
    print('query points: {}'.format(_describe_counts(
        counts, roi=args.roi is not None, dark=min_intensity > 0,
        visible=args.visible_only)))
    print('matches with std_f < {}: {}'.format(args.th, len(mkpts0)))

    if args.ransac and len(mkpts0) > 0:
        if data['pose1'] is None or data['pose2'] is None:
            raise ValueError('--ransac needs --pose1 and --pose2')
        keep = ransac_outlier_rejection(
            mkpts0, mkpts1, data['pose1'], data['pose2'], data['meta1'], data['meta2'],
            num_iterations=args.ransac_iters, threshold=args.ransac_th,
            rng=np.random.default_rng(args.seed))
        mkpts0, mkpts1, std_f = mkpts0[keep], mkpts1[keep], std_f[keep]
        print('matches after RANSAC: {}'.format(len(mkpts0)))

    if data['pose1'] is not None and data['pose2'] is not None and len(mkpts0) > 0:
        projected = project_keypoints(mkpts0, data['pose1'], data['pose2'],
                                      data['meta1'], data['meta2'])
        _, percentage, mean_dist, std_dist = inlier_percentage(
            projected, mkpts1, px_th=args.px_th)
        print('inliers within {:g} px: {:.1f}%'.format(args.px_th, percentage))
        print('mean pixel error: {:.2f} px (std {:.2f})'.format(mean_dist, std_dist))

    print('elapsed: {:.2f} s'.format(time.time() - started))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    make_matching_figure(data['im1'], data['im2'], mkpts0, mkpts1, path=args.out)
    print('saved {}'.format(args.out))


if __name__ == '__main__':
    main()
