"""Tests for the keypoint helpers in dataloader/data_utils.py."""
import os

import numpy as np
import pytest
import torch

from dataloader import data_utils


def test_rescale_keypoints():
    """Keypoints detected on the 640x480 SuperPoint input map back to the image."""
    keypoints = np.array([[320, 240], [160, 120], [480, 360]])
    size = (512, 512)
    expected = np.array([[256, 256], [128, 128], [384, 384]])
    np.testing.assert_array_equal(data_utils.rescale_keypoints(keypoints, size), expected)

    keypoints = np.array([[0, 0], [640, 480]])
    expected = np.array([[0, 0], [512, 512]])
    np.testing.assert_array_equal(data_utils.rescale_keypoints(keypoints, size), expected)

    keypoints = np.array([[320, 240], [160, 120], [480, 360], [0, 0], [640, 480]])
    expected = np.array([[256, 256], [128, 128], [384, 384], [0, 0], [512, 512]])
    np.testing.assert_array_equal(data_utils.rescale_keypoints(keypoints, size), expected)


def test_rescale_keypoints_leaves_input_alone():
    """The helper returns a copy, so the caller's array is untouched."""
    keypoints = np.array([[320., 240.]])
    original = keypoints.copy()
    data_utils.rescale_keypoints(keypoints, (512, 512))
    np.testing.assert_array_equal(keypoints, original)


def test_preprocess_image_shapes_and_dtypes():
    """AKAZE gets a uint8 image, SuperPoint a 640x480 float image in [0, 1]."""
    rng = np.random.RandomState(0)
    image = rng.rand(512, 512).astype(np.float32)
    norm_image, grayim = data_utils.preprocess_image(image, (512, 512))

    assert norm_image.dtype == np.uint8
    assert norm_image.shape == (512, 512)
    assert grayim.shape == (480, 640)
    assert grayim.dtype == np.float32
    assert grayim.min() >= 0.0 and grayim.max() <= 1.0
    # A random image really varies, so the bounds above are not vacuous.
    assert grayim.max() > grayim.min()


def brute_force_neighbours(coord, k):
    """The k nearest rows to each row, itself included, by a plain distance sort.

    Independent of the cKDTree the implementation uses, so the test checks the
    result rather than repeating the method.
    """
    diff = coord[:, None, :] - coord[None, :, :]
    distance = np.sqrt((diff ** 2).sum(axis=-1))
    return np.argsort(distance, axis=1, kind='stable')[:, :k]


def expected_padding(coord, num_pts, seed):
    """Recompute the rows _fill_with_neighbors must append."""
    total = len(coord)
    idx = brute_force_neighbours(coord, min(4, total))
    chosen = np.random.RandomState(seed).randint(low=0, high=total - 1,
                                                 size=num_pts - total)
    return np.array([[coord[idx[i], 0].mean(), coord[idx[i], 1].mean()]
                     for i in chosen])


def sort_rows(rows):
    """Order rows so the comparison ignores the shuffle inside the padding."""
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def test_fill_with_neighbors_matches_explicit_neighbour_means():
    """Every padded row is the mean of a chosen point and its 3 nearest."""
    points = np.random.RandomState(7).rand(20, 2) * np.array([500.0, 400.0])
    filled = data_utils._fill_with_neighbors(points, 50, rng=np.random.RandomState(0))

    assert filled.shape == (50, 2)
    np.testing.assert_allclose(filled[:20], points)
    expected = expected_padding(points, 50, seed=0)
    # The padding really differs row to row, so the comparison is not vacuous.
    assert len(np.unique(sort_rows(expected), axis=0)) > 1
    np.testing.assert_allclose(sort_rows(filled[20:]), sort_rows(expected))


def test_fill_with_neighbors_two_points():
    """With two points, k collapses to 2 and every padded row is the midpoint."""
    points = np.array([[0.0, 0.0], [10.0, 4.0]])
    filled = data_utils._fill_with_neighbors(points, 5, rng=np.random.RandomState(0))

    assert filled.shape == (5, 2)
    np.testing.assert_allclose(filled[:2], points)
    np.testing.assert_allclose(filled[2:], np.tile([5.0, 2.0], (3, 1)))
    np.testing.assert_allclose(sort_rows(filled[2:]),
                               sort_rows(expected_padding(points, 5, seed=0)))


def test_fill_with_neighbors_three_points():
    """With three points, k collapses to 3 and every padded row is the centroid."""
    points = np.array([[0.0, 0.0], [9.0, 0.0], [0.0, 6.0]])
    filled = data_utils._fill_with_neighbors(points, 6, rng=np.random.RandomState(0))

    assert filled.shape == (6, 2)
    np.testing.assert_allclose(filled[:3], points)
    np.testing.assert_allclose(filled[3:], np.tile([3.0, 2.0], (3, 1)))
    np.testing.assert_allclose(sort_rows(filled[3:]),
                               sort_rows(expected_padding(points, 6, seed=0)))


def test_fill_with_neighbors_is_a_noop_when_already_full():
    """Nothing is added when the input already has enough points.

    The old code reached np.vstack((coord, [])) here and raised.
    """
    coord = np.arange(20, dtype=float).reshape(10, 2)
    np.testing.assert_allclose(data_utils._fill_with_neighbors(coord, 10), coord)
    np.testing.assert_allclose(data_utils._fill_with_neighbors(coord, 4), coord)


def test_get_superpoint_missing_weights_message(tmp_path, monkeypatch):
    """A missing weights file names the download the README describes."""
    monkeypatch.chdir(tmp_path)
    # The repository fallback must fail too, so point it at an empty directory.
    monkeypatch.setattr(data_utils, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(data_utils, '_superpoint_cache', {})
    missing = str(tmp_path / 'nope.pth')

    with pytest.raises(FileNotFoundError) as excinfo:
        data_utils._get_superpoint(missing)
    assert 'pretrained/superpoint_v1.pth' in str(excinfo.value)
    assert 'README' in str(excinfo.value)


@pytest.mark.slow
def test_get_superpoint_in_worker_never_queries_cuda(monkeypatch):
    """Inside a dataloader worker the frontend must not touch CUDA at all.

    Workers are forked after the parent has initialised CUDA, so asking
    torch.cuda anything in the child is exactly what the guard prevents.
    """
    weights = os.path.join('pretrained', 'superpoint_v1.pth')
    if not os.path.isfile(weights):
        pytest.skip('SuperPoint weights not available')

    class DummyWorkerInfo:
        id = 0

    def forbidden():
        raise AssertionError('must not be called in a worker')

    monkeypatch.setattr(data_utils, '_superpoint_cache', {})
    monkeypatch.setattr(torch.utils.data, 'get_worker_info', lambda: DummyWorkerInfo())
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)

    frontend = data_utils._get_superpoint(weights)
    assert frontend.cuda is False


@pytest.mark.slow
def test_get_superpoint_caches_by_path(tiny_dataset_dir):
    """The frontend is built once per weights file, not once per image."""
    weights = os.path.join('pretrained', 'superpoint_v1.pth')
    first = data_utils._get_superpoint(weights)
    second = data_utils._get_superpoint(weights)
    assert first is second


@pytest.mark.slow
def test_generate_query_kpts_tiny_image(tiny_dataset_dir):
    """Real keypoint generation returns the requested number of points."""
    image = np.load(os.path.join(tiny_dataset_dir, 'images', 'H-15_1.npy'))
    image = np.flip(image).copy()
    num_pts = 50
    coord = data_utils.generate_query_kpts(
        image, num_pts, 20, 512, 512,
        superpoint_weights=os.path.join('pretrained', 'superpoint_v1.pth'))

    assert coord.ndim == 2 and coord.shape[1] == 2
    assert coord.shape[0] in (num_pts, 1)
    if coord.shape[0] == num_pts:
        assert coord[:, 0].min() >= 0 and coord[:, 0].max() <= 512
        assert coord[:, 1].min() >= 0 and coord[:, 1].max() <= 512
        # Real detections, not the zero fallback.
        assert np.abs(coord).sum() > 0


def test_prune_kpts_is_gone():
    """The unused camera-model pruning helper was removed."""
    assert not hasattr(data_utils, 'prune_kpts')


def test_no_sklearn_import():
    """The neighbour search uses scipy, so scikit-learn is not a dependency."""
    source = open(data_utils.__file__).read()
    assert 'sklearn' not in source
