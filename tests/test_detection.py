"""Tests for detection.py and for the geometry helpers in CSONIC_test.py."""
import os

import cv2
import numpy as np
import pytest

import CSONIC_test
import detection

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERPOINT_WEIGHTS = os.path.join(REPO_ROOT, 'pretrained', 'superpoint_v1.pth')

# The real-world sample pairs. Set CSONIC_SAMPLE_DIR to the directory holding
# them; the tests that need real sonar images skip when it is not set.
SAMPLE_DIR = os.environ.get('CSONIC_SAMPLE_DIR')

# The metadata the real-world sonar images carry.
META_H = {'width': 512, 'height': 512, 'r_min': 0.1, 'r_max': 7, 'elev': 12, 'azi': 60}
META_L = {'width': 512, 'height': 512, 'r_min': 0.1, 'r_max': 7, 'elev': 20, 'azi': 130}


def _sample(rel):
    """Return the path of a real-world sample, or skip when it is missing."""
    if not SAMPLE_DIR:
        pytest.skip('set CSONIC_SAMPLE_DIR to the directory holding the sonar '
                    'samples to run this test')
    path = os.path.join(SAMPLE_DIR, rel)
    if not os.path.isfile(path):
        pytest.skip('sample not found: {}; check CSONIC_SAMPLE_DIR'.format(path))
    return path


def _need_superpoint():
    if not os.path.isfile(SUPERPOINT_WEIGHTS):
        pytest.skip('SuperPoint weights not found at {}'.format(SUPERPOINT_WEIGHTS))


def _released_preprocess(image):
    """The sonar branch of sonic's ``detection.preprocess_image``, transcribed.

    The first two assignments are dead stores in the released script: the
    anisotropic diffusion result and the min-max normalised image are both
    overwritten, because every step reads ``image`` again. The denoised raw
    image is what reaches the unsharp mask, and that is the pipeline the
    released results came from.
    """
    norm_image = cv2.normalize(image, None, alpha=0, beta=255,
                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    norm_image = detection.anisotropic_diffusion(image, iterations=2, delta_t=0.08, kappa=10)
    norm_image = cv2.fastNlMeansDenoising(image, None, 5, 7, 21)
    norm_image = detection.unsharp_mask(norm_image, kernel_size=(5, 5), sigma=0.5,
                                        amount=2.0, threshold=0.0)
    return norm_image.astype(np.uint8)


# --------------------------------------------------------------------------
# detection.load_image / load_data
# --------------------------------------------------------------------------

def test_load_image_resizes_to_meta_and_flips(tmp_path):
    """A 619x512 PNG becomes 512x512, and the bright block moves to the far corner."""
    raw = np.zeros((619, 512), dtype=np.uint8)
    raw[96:112, 96:112] = 255  # a block in the top left quadrant
    png = str(tmp_path / 'sonar.png')
    assert cv2.imwrite(png, raw)

    out = detection.load_image(png, META_H)

    assert out.shape == (512, 512)
    assert out.dtype == np.uint8
    row, col = np.unravel_index(int(np.argmax(out)), out.shape)
    # np.flip reverses both axes, so the block ends up in the bottom right.
    assert row > 256 and col > 256
    assert out[:256, :256].max() == 0
    assert out.max() > 128  # the block survived the resize


def test_load_image_png_without_meta_keeps_its_shape(tmp_path):
    """Without metadata the image is returned at the size it was stored at."""
    raw = np.zeros((619, 512), dtype=np.uint8)
    raw[300:320, 40:60] = 200
    png = str(tmp_path / 'sonar.png')
    assert cv2.imwrite(png, raw)

    out = detection.load_image(png)

    assert out.shape == (619, 512)
    assert np.array_equal(out, np.flip(raw))


def test_load_image_npy_is_min_max_scaled_to_uint8(tmp_path):
    """A float .npy is scaled into the uint8 range in memory, and flipped."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(64, 80)) * 3.0 + 1.0
    npy = str(tmp_path / 'sonar.npy')
    np.save(npy, arr)

    out = detection.load_image(npy)

    assert out.dtype == np.uint8
    assert out.shape == (64, 80)
    assert out.min() == 0 and out.max() == 255
    scaled = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
    assert np.array_equal(out, np.flip(scaled))
    # No scratch file is left behind; the released script wrote s1.png / s2.png.
    assert sorted(p.name for p in tmp_path.iterdir()) == ['sonar.npy']


def test_load_image_npy_constant_array_is_all_zero(tmp_path):
    """A constant image has no range to scale, so it comes back as zeros."""
    npy = str(tmp_path / 'flat.npy')
    np.save(npy, np.full((8, 9), 4.25))

    out = detection.load_image(npy)

    assert out.shape == (8, 9)
    assert out.dtype == np.uint8
    assert out.max() == 0


def test_load_data_fills_the_dict_and_leaves_the_rest_none(tmp_path):
    """Poses and metadata are optional; the keys are always present."""
    raw = np.zeros((619, 512), dtype=np.uint8)
    raw[200:260, 200:260] = 180
    png1 = str(tmp_path / 'a.png')
    png2 = str(tmp_path / 'b.png')
    assert cv2.imwrite(png1, raw)
    assert cv2.imwrite(png2, np.flip(raw))

    data = detection.load_data(png1, png2)
    assert set(data) == {'im1', 'im2', 'pose1', 'pose2', 'meta1', 'meta2'}
    assert data['im1'].shape == (619, 512)
    assert data['pose1'] is None and data['pose2'] is None
    assert data['meta1'] is None and data['meta2'] is None

    pose = np.eye(4)
    pose[0, 3] = 1.5
    pose_path = str(tmp_path / 'p.npy')
    np.save(pose_path, pose)
    meta_path = str(tmp_path / 'm.npy')
    np.save(meta_path, META_H)

    data = detection.load_data(png1, png2, pose_path, pose_path, meta_path, meta_path)
    assert np.array_equal(data['pose1'], pose)
    assert data['meta2'] == META_H
    # With metadata the images are resized to the metadata size.
    assert data['im1'].shape == (META_H['height'], META_H['width'])


def test_read_meta_returns_a_dict(tmp_path):
    meta_path = str(tmp_path / 'm.npy')
    np.save(meta_path, META_L)

    meta = detection.read_meta(meta_path)

    assert isinstance(meta, dict)
    assert meta['azi'] == 130 and meta['r_max'] == 7


# --------------------------------------------------------------------------
# detection.preprocess_image
# --------------------------------------------------------------------------

def test_preprocess_image_shapes():
    """AKAZE gets the image at its own size; SuperPoint gets 480x640 in [0, 1]."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)

    norm_image, grayim = detection.preprocess_image(img)

    assert norm_image.dtype == np.uint8
    assert norm_image.shape == img.shape
    assert grayim.dtype == np.float32
    assert grayim.shape == (480, 640)
    assert grayim.min() >= 0.0 and grayim.max() <= 1.0
    # The pipeline has to change the image, or it is doing nothing.
    assert not np.array_equal(norm_image, img)


def test_preprocess_image_matches_the_released_pipeline():
    """The output is the same image the sonic eval script produced."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(128, 160), dtype=np.uint8)

    norm_image, _ = detection.preprocess_image(img)

    assert np.array_equal(norm_image, _released_preprocess(img))


def test_anisotropic_diffusion_smooths_and_stays_uint8():
    """The helper returns a uint8 image with less high frequency content."""
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

    out = detection.anisotropic_diffusion(img, iterations=2, delta_t=0.08, kappa=10)

    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert np.abs(np.diff(out.astype(float), axis=1)).mean() \
        < np.abs(np.diff(img.astype(float), axis=1)).mean()


# --------------------------------------------------------------------------
# detection.generate_query_kpts
# --------------------------------------------------------------------------

def test_generate_query_kpts_real_image():
    """Both detectors fire on a real sonar image and the points stay in frame."""
    _need_superpoint()
    img = detection.load_image(_sample('H0/3.png'), META_H)

    coord, coord_a, coord_s, desc_a, desc_s = detection.generate_query_kpts(
        img, h=META_H['height'], w=META_H['width'],
        superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(coord_a) > 0 and len(coord_s) > 0
    assert len(coord) == len(coord_a) + len(coord_s)
    assert coord.shape[1] == 2
    assert coord[:, 0].min() >= 0 and coord[:, 0].max() < META_H['width']
    assert coord[:, 1].min() >= 0 and coord[:, 1].max() < META_H['height']
    # AKAZE descriptors are one row per keypoint; SuperPoint gives 256 x N.
    assert desc_a.shape[0] == len(coord_a)
    assert desc_s.shape == (256, len(coord_s))


def test_generate_query_kpts_blank_image_returns_superpoint_only():
    """AKAZE finds nothing on a flat image, so only SuperPoint points come back."""
    _need_superpoint()
    img = np.zeros((512, 512), dtype=np.uint8)

    coord, coord_a, coord_s, desc_a, _ = detection.generate_query_kpts(
        img, h=512, w=512, superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(coord_a) == 0
    assert len(coord) == len(coord_s)
    # An empty detection still has to be an (N, 2) array, or the caller's
    # column indexing fails.
    assert coord_a.shape == (0, 2)
    assert coord.ndim == 2 and coord.shape[1] == 2
    assert desc_a is None or len(desc_a) == 0


# --------------------------------------------------------------------------
# CSONIC_test geometry helpers
# --------------------------------------------------------------------------

def test_geometry_roundtrip():
    """Pixels survive the trip through polar coordinates and back."""
    pix = np.array([[0.0, 0.0], [511.0, 511.0], [256.0, 128.0], [10.5, 480.25],
                    [300.0, 12.0]])
    geom = dict(image_width=META_H['width'], image_height=META_H['height'],
                bearing_max=META_H['azi'], range_max=META_H['r_max'],
                range_min=META_H['r_min'])

    bearing, rng_m = CSONIC_test.pix_to_polar_batch(pix, **geom)

    # The polar values have to be physically sensible, not just invertible.
    half_fov = np.deg2rad(META_H['azi']) / 2
    assert np.all(np.abs(bearing) <= half_fov + 1e-9)
    assert np.all(rng_m >= META_H['r_min'] - 1e-9)
    assert np.all(rng_m <= META_H['r_max'] + 1e-9)

    u, v = CSONIC_test.polar_to_pix_batch((bearing, rng_m), **geom)

    assert np.allclose(np.stack([u, v], axis=1), pix, atol=1e-6)


def test_geometry_bearing_and_range_have_the_expected_signs():
    """The sensor sits at the bottom centre: near range is the bottom row."""
    geom = dict(image_width=512, image_height=512, bearing_max=60.0,
                range_max=7.0, range_min=0.1)
    pix = np.array([[256.0, 511.0], [256.0, 0.0], [0.0, 256.0], [511.0, 256.0]])

    bearing, rng_m = CSONIC_test.pix_to_polar_batch(pix, **geom)

    assert rng_m[0] < rng_m[1]           # the bottom row is the near range
    assert abs(bearing[0]) < 1e-3        # the centre column is bearing zero
    assert bearing[2] > 0 > bearing[3]   # column 0 is a positive bearing


def test_polar_to_cartesian_sampled_points_matches_the_training_criterion():
    """The numpy arc sampler agrees with CtoFCriterion.convert_to_arc."""
    torch = pytest.importorskip('torch')
    from CSONIC.criterion import CtoFCriterion

    rng_m = np.array([1.0, 4.5, 6.75])
    bearing = np.array([-0.4, 0.0, 0.3])
    phi_deg = 12.0

    got = CSONIC_test.polar_to_cartesian_sampled_points(rng_m, bearing, phi_deg,
                                                        sample_points=5)

    want = CtoFCriterion.convert_to_arc(
        torch.tensor(rng_m).float()[None],
        torch.tensor(bearing).float()[None],
        CtoFCriterion.batch_linspace(torch.tensor([-phi_deg / 2]),
                                     torch.tensor([phi_deg / 2]), 5),
    ).numpy()[0]

    # got is (3, n_pts, n_samples); want is (n_pts, n_samples, 3).
    assert np.allclose(np.moveaxis(got, 0, -1), want, atol=1e-5)


def test_project_keypoints_identity_pose():
    """With the same pose for both images the projection is the identity."""
    mkpts0 = np.array([[100.0, 200.0], [256.0, 300.0], [400.0, 450.0]])
    pose = np.eye(4)
    pose[:3, :3] = np.array([[0.8660254, 0.0, 0.5],
                             [0.0, 1.0, 0.0],
                             [-0.5, 0.0, 0.8660254]])
    pose[:3, 3] = [-4.88, -6.48, -4.48]

    pix = CSONIC_test.project_keypoints(mkpts0, pose, pose, META_H, META_H)

    assert pix.shape == mkpts0.shape
    assert np.allclose(pix, mkpts0, atol=1e-6)


def test_project_keypoints_translation_shifts_pixels():
    """A real baseline has to move the pixels; otherwise the pose is ignored."""
    mkpts0 = np.array([[100.0, 200.0], [256.0, 300.0], [400.0, 450.0]])
    pose1 = np.eye(4)
    pose2 = np.eye(4)
    pose2[0, 3] = 1.2  # 1.2 m along x

    pix = CSONIC_test.project_keypoints(mkpts0, pose1, pose2, META_H, META_H)

    assert pix.shape == mkpts0.shape
    assert np.all(np.isfinite(pix))
    assert np.linalg.norm(pix - mkpts0, axis=1).min() > 1.0


def test_project_keypoints_uses_the_second_metadata():
    """The output pixels are laid out with image 2's field of view."""
    mkpts0 = np.array([[100.0, 200.0], [256.0, 300.0], [400.0, 450.0]])
    pose = np.eye(4)

    same = CSONIC_test.project_keypoints(mkpts0, pose, pose, META_H, META_H)
    wider = CSONIC_test.project_keypoints(mkpts0, pose, pose, META_H, META_L)

    # META_L has a 130 degree azimuth, so the same bearing lands nearer the centre.
    assert not np.allclose(same[:, 0], wider[:, 0])
    assert np.allclose(same[:, 1], wider[:, 1], atol=1e-6)  # the ranges agree


def _rot_z(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _asymmetric_pose_pair():
    """Two extrinsics whose relative pose is not its own inverse.

    A symmetric pair would let a reversed ``T2 @ inv(T1)`` pass unnoticed.
    """
    relative = np.eye(4)
    relative[:3, :3] = _rot_z(7.0) @ _rot_y(3.0)
    relative[:3, 3] = [0.4, -0.2, 0.3]
    first = np.eye(4)
    first[:3, :3] = _rot_y(11.0)
    first[:3, 3] = [0.15, 0.25, -0.05]
    return first, relative @ first


def test_arc_transform_matches_the_training_criterion():
    """The numpy arc path agrees with the torch path the model trained with.

    This pins the direction of the relative pose as well as the arithmetic: the
    pair is asymmetric, so swapping T1 and T2 cannot give the same answer.
    """
    torch = pytest.importorskip('torch')
    from CSONIC.criterion import CtoFCriterion

    T1, T2 = _asymmetric_pose_pair()
    mkpts0 = np.array([[120.0, 260.0], [300.0, 380.0], [420.0, 300.0]])
    n_samples = 7
    elev = float(META_H['elev'])

    bearing, ranges = CSONIC_test.pix_to_polar_batch(
        mkpts0, range_max=META_H['r_max'], range_min=META_H['r_min'],
        image_width=META_H['width'], image_height=META_H['height'],
        bearing_max=META_H['azi'])
    arc = CSONIC_test.polar_to_cartesian_sampled_points(ranges, bearing, elev, n_samples)
    moved = CSONIC_test.transform_points(arc, T1, T2)
    got_bearing, got_range = CSONIC_test.cart_to_polar_batch(moved)

    rot, translation = CtoFCriterion.get_rel_pose(
        torch.tensor(T1).float().unsqueeze(0), torch.tensor(T2).float().unsqueeze(0))
    phi = CtoFCriterion.batch_linspace(
        torch.tensor([-elev / 2.0]), torch.tensor([elev / 2.0]), n_samples)
    sampled = CtoFCriterion.convert_to_arc(
        torch.tensor(ranges).float().unsqueeze(0),
        torch.tensor(bearing).float().unsqueeze(0), phi)
    want_moved = CtoFCriterion.transform_arc_points(sampled, rot, translation)
    want_bearing, want_range = CtoFCriterion.cartesian_arc_to_polar(want_moved)

    assert np.allclose(np.moveaxis(moved, 0, -1), want_moved.numpy()[0], atol=1e-5)
    assert np.allclose(got_bearing, want_bearing.numpy()[0], atol=1e-5)
    assert np.allclose(got_range, want_range.numpy()[0], atol=1e-5)

    # The same call with the poses swapped must not agree, or the test above
    # would pass on a reversed relative pose.
    reversed_moved = CSONIC_test.transform_points(arc, T2, T1)
    assert not np.allclose(np.moveaxis(reversed_moved, 0, -1),
                           want_moved.numpy()[0], atol=1e-3)


def test_project_keypoints_uses_the_criterion_pose_direction():
    """project_keypoints lands where the training criterion says it should."""
    torch = pytest.importorskip('torch')
    from CSONIC.criterion import CtoFCriterion

    pose1, pose2 = _asymmetric_pose_pair()
    mkpts0 = np.array([[120.0, 260.0], [300.0, 380.0], [420.0, 300.0]])

    pix = CSONIC_test.project_keypoints(mkpts0, pose1, pose2, META_H, META_L)

    T1 = torch.tensor(CSONIC_test.pose_correction(pose1)).float().unsqueeze(0)
    T2 = torch.tensor(CSONIC_test.pose_correction(pose2)).float().unsqueeze(0)
    bearing, ranges = CSONIC_test.pix_to_polar_batch(
        mkpts0, range_max=META_H['r_max'], range_min=META_H['r_min'],
        image_width=META_H['width'], image_height=META_H['height'],
        bearing_max=META_H['azi'])
    rot, translation = CtoFCriterion.get_rel_pose(T1, T2)
    # project_keypoints takes the centre of the arc, elevation zero.
    phi = CtoFCriterion.batch_linspace(torch.tensor([0.0]), torch.tensor([0.0]), 1)
    sampled = CtoFCriterion.convert_to_arc(
        torch.tensor(ranges).float().unsqueeze(0),
        torch.tensor(bearing).float().unsqueeze(0), phi)
    moved = CtoFCriterion.transform_arc_points(sampled, rot, translation)
    want_bearing, want_range = CtoFCriterion.cartesian_arc_to_polar(moved)
    u, v = CtoFCriterion.polar_to_pix(
        (want_bearing, want_range), image_width=float(META_L['width']),
        image_height=float(META_L['height']), bearing_max=float(META_L['azi']),
        range_max=float(META_L['r_max']), range_min=float(META_L['r_min']))

    want = np.stack([u.numpy().ravel(), v.numpy().ravel()], axis=1)
    assert np.allclose(pix, want, atol=1e-3)


def test_sonar_epipolar_cost_is_small_for_a_true_correspondence():
    """A projected point costs far less than a point 100 px away."""
    mkpts0 = np.array([[120.0, 260.0], [300.0, 380.0], [420.0, 300.0]])
    pose1 = np.eye(4)
    pose2 = np.eye(4)
    pose2[0, 3] = 0.4
    truth = CSONIC_test.project_keypoints(mkpts0, pose1, pose2, META_H, META_H)
    # The cost works in extrinsics; project_keypoints takes the raw poses.
    T1 = CSONIC_test.pose_correction(pose1)
    T2 = CSONIC_test.pose_correction(pose2)

    good = CSONIC_test.sonar_epipolar_cost_np(mkpts0, truth, META_H, META_H, T1, T2)
    bad = CSONIC_test.sonar_epipolar_cost_np(mkpts0, truth + 100.0, META_H, META_H,
                                             T1, T2)

    assert good.shape == (len(mkpts0),)
    assert np.all(good < 1e-3)
    assert np.all(bad > good + 1e-2)


def test_ransac_outlier_rejection_keeps_the_true_matches():
    """The inlier set holds the projected matches and drops the scattered ones."""
    rng = np.random.default_rng(4)
    mkpts0 = np.column_stack([rng.uniform(60, 450, 40), rng.uniform(150, 460, 40)])
    pose1 = np.eye(4)
    pose2 = np.eye(4)
    pose2[0, 3] = 0.35
    mkpts1 = CSONIC_test.project_keypoints(mkpts0, pose1, pose2, META_H, META_H)
    # Corrupt the second half with large, random errors.
    mkpts1[20:] += rng.uniform(90, 150, size=(20, 2)) * rng.choice([-1.0, 1.0], size=(20, 2))

    keep = CSONIC_test.ransac_outlier_rejection(
        mkpts0, mkpts1, pose1, pose2, META_H, META_H, num_iterations=100,
        threshold=0.03, rng=np.random.default_rng(5))

    keep = np.asarray(keep)
    assert len(keep) >= 15
    assert (keep < 20).mean() > 0.9  # almost every kept match is a true one


@pytest.mark.parametrize('n', [0, 1, 3])
def test_ransac_handles_tiny_match_sets(n):
    """Fewer matches than the quarter it samples must not crash or over-reach."""
    rng = np.random.default_rng(11)
    mkpts0 = np.column_stack([rng.uniform(60, 450, n), rng.uniform(150, 460, n)])
    pose1, pose2 = np.eye(4), np.eye(4)
    pose2[0, 3] = 0.3
    mkpts1 = CSONIC_test.project_keypoints(mkpts0, pose1, pose2, META_H, META_H)

    keep = np.asarray(CSONIC_test.ransac_outlier_rejection(
        mkpts0, mkpts1, pose1, pose2, META_H, META_H, num_iterations=20,
        rng=np.random.default_rng(0)))

    assert keep.ndim == 1 and keep.dtype.kind == 'i'
    assert set(keep.tolist()) <= set(range(n))


def test_ransac_keeps_everything_when_no_match_costs_anything(monkeypatch):
    """A set that all sits exactly on its arc has nothing to reject."""
    monkeypatch.setattr(CSONIC_test, 'sonar_epipolar_cost_np',
                        lambda *args, **kwargs: np.zeros(5))

    keep = np.asarray(CSONIC_test.ransac_outlier_rejection(
        np.zeros((5, 2)), np.zeros((5, 2)), np.eye(4), np.eye(4), META_H, META_H,
        num_iterations=10, rng=np.random.default_rng(0)))

    assert np.array_equal(keep, np.arange(5))


def test_ransac_keeps_nothing_when_no_cost_is_finite(monkeypatch):
    """An unusable cost is not the same as a cost of zero.

    Keeping every match because none could be scored would be a claim the data
    does not support.
    """
    monkeypatch.setattr(CSONIC_test, 'sonar_epipolar_cost_np',
                        lambda *args, **kwargs: np.full(5, np.nan))

    keep = np.asarray(CSONIC_test.ransac_outlier_rejection(
        np.zeros((5, 2)), np.zeros((5, 2)), np.eye(4), np.eye(4), META_H, META_H,
        num_iterations=10, rng=np.random.default_rng(0)))

    assert keep.dtype.kind == 'i'
    assert len(keep) == 0


def test_ransac_with_a_broken_pose_keeps_nothing():
    """The same, reached the way it happens in practice: a pose with a NaN."""
    rng = np.random.default_rng(12)
    mkpts0 = np.column_stack([rng.uniform(60, 450, 8), rng.uniform(150, 460, 8)])
    pose1, pose2 = np.eye(4), np.eye(4)
    pose2[0, 3] = np.nan

    keep = np.asarray(CSONIC_test.ransac_outlier_rejection(
        mkpts0, mkpts0.copy(), pose1, pose2, META_H, META_H, num_iterations=20,
        rng=np.random.default_rng(0)))

    assert len(keep) == 0


def test_ransac_outlier_rejection_is_reproducible_with_a_generator():
    rng_pts = np.random.default_rng(6)
    mkpts0 = np.column_stack([rng_pts.uniform(60, 450, 30), rng_pts.uniform(150, 460, 30)])
    pose1, pose2 = np.eye(4), np.eye(4)
    pose2[1, 3] = 0.25
    mkpts1 = CSONIC_test.project_keypoints(mkpts0, pose1, pose2, META_H, META_H)
    mkpts1[15:] += 120.0

    args = (mkpts0, mkpts1, pose1, pose2, META_H, META_H)
    first = CSONIC_test.ransac_outlier_rejection(
        *args, num_iterations=50, threshold=0.03, rng=np.random.default_rng(7))
    second = CSONIC_test.ransac_outlier_rejection(
        *args, num_iterations=50, threshold=0.03, rng=np.random.default_rng(7))

    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_inlier_percentage_counts_only_the_close_matches():
    gt = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    pred = gt.copy()
    pred[2:] += 60.0  # two matches well outside a 20 px threshold

    idx, percentage, mean_dist, std_dist = CSONIC_test.inlier_percentage(gt, pred, px_th=20)

    assert np.array_equal(np.asarray(idx), np.array([0, 1]))
    assert percentage == pytest.approx(50.0)
    dists = np.linalg.norm(pred - gt, axis=1)
    assert mean_dist == pytest.approx(dists.mean())
    assert std_dist == pytest.approx(dists.std())


def test_expectation_matching_needs_a_model():
    """Neither a loaded model nor a checkpoint path is an error, not a crash."""
    with pytest.raises(ValueError):
        CSONIC_test.expectation_matching('a.png', 'b.png', 'a_meta.npy', 'b_meta.npy')


def test_parse_args_reads_the_flags():
    args = CSONIC_test.parse_args([
        '--img1', 'a.png', '--img2', 'b.png',
        '--meta1', 'a_meta.npy', '--meta2', 'b_meta.npy',
        '--model', 'ckpt.pth', '--th', '0.2', '--ransac', '--ransac-iters', '50',
        '--roi', '0', '512', '220', '400', '--legacy-input', '--bn', 'running'])

    assert args.th == 0.2
    assert args.ransac is True and args.ransac_iters == 50
    assert args.roi == [0.0, 512.0, 220.0, 400.0]
    assert args.legacy_input is True
    assert args.bn == 'running'
    # The defaults the report was produced with.
    assert args.ransac_th == 0.1 and args.px_th == 20.0 and args.seed == 0


def test_parse_args_bn_defaults_to_the_library_default():
    args = CSONIC_test.parse_args([
        '--img1', 'a.png', '--img2', 'b.png',
        '--meta1', 'a_meta.npy', '--meta2', 'b_meta.npy', '--model', 'ckpt.pth'])

    assert args.bn == CSONIC_test.DEFAULT_BN_MODE


def test_parse_args_rejects_an_unknown_bn_mode():
    with pytest.raises(SystemExit):
        CSONIC_test.parse_args([
            '--img1', 'a.png', '--img2', 'b.png', '--meta1', 'a_meta.npy',
            '--meta2', 'b_meta.npy', '--model', 'ckpt.pth', '--bn', 'sideways'])


def test_match_points_rejects_an_unknown_bn_mode():
    """The mode is checked before anything else, so a typo fails at once."""
    blank = np.zeros((8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match='bn_mode'):
        CSONIC_test.match_points(None, blank, blank, np.zeros((1, 2)),
                                 bn_mode='sideways')


def test_default_bn_mode_is_pinned():
    """The default was chosen from the measurements recorded in results.md.

    Changing it changes every number a user gets, so it is pinned here.
    """
    assert CSONIC_test.DEFAULT_BN_MODE == 'batch'
    assert CSONIC_test.BN_MODES == ('running', 'batch')


def test_inlier_percentage_pairs_rows_not_nearest_neighbours():
    """Row i of the prediction is compared with row i of the truth."""
    gt = np.array([[10.0, 10.0], [400.0, 400.0]])
    pred = gt[::-1].copy()  # the same points, in the wrong order

    idx, percentage, _, _ = CSONIC_test.inlier_percentage(gt, pred, px_th=20)

    assert len(idx) == 0
    assert percentage == pytest.approx(0.0)
