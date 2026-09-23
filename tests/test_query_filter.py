"""Tests for the query point filters in CSONIC_test.py.

Two filters decide which detected points the network is asked about:

* the brightness filter drops a point that sits in a dark part of image 1,
  where there is no return to match. It is on by default;
* the view filter uses the two poses to drop a point that sonar 2 cannot see.
  It is off by default.

Nothing here needs the SuperPoint weights or the checkpoint: the detector, the
data loader and the network are replaced where a test would reach them.
"""
import inspect

import numpy as np
import pytest

import CSONIC_test
import detection

# The metadata of the released sample: low frequency and high frequency.
META_L = {'width': 512, 'height': 512, 'r_min': 0.1, 'r_max': 10, 'elev': 20, 'azi': 130}
META_H = {'width': 512, 'height': 512, 'r_min': 0.1, 'r_max': 10, 'elev': 12, 'azi': 60}


def _rot_x(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(degrees):
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose(rotation=None, translation=(0.0, 0.0, 0.0)):
    pose = np.eye(4)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _world_pose():
    """A sensor-to-world pose that is not the identity, like a real one."""
    return _pose(_rot_z(-90.0) @ _rot_y(15.0), (6.76, -12.13, -3.18))


def _moved(pose1, rotation=None, translation=(0.0, 0.0, 0.0)):
    """The pose of a sensor moved by ``rotation`` and ``translation`` in its own frame."""
    return pose1 @ _pose(rotation, translation)


def _pix(bearing_deg, range_m, meta):
    """The pixels of image ``meta`` at these bearings (degrees) and ranges (m)."""
    bearing = np.deg2rad(np.asarray(bearing_deg, dtype=float))
    ranges = np.broadcast_to(np.asarray(range_m, dtype=float), bearing.shape)
    u, v = CSONIC_test.polar_to_pix_batch(
        (bearing, ranges), image_width=meta['width'], image_height=meta['height'],
        bearing_max=meta['azi'], range_max=meta['r_max'], range_min=meta['r_min'])
    return np.stack([np.ravel(u), np.ravel(v)], axis=1)


def _half_bright(shape=(128, 128), value=60):
    """A uint8 image: the top half at ``value``, the bottom half black."""
    img = np.zeros(shape, dtype=np.uint8)
    img[:shape[0] // 2] = value
    return img


# --------------------------------------------------------------------------
# the brightness filter
# --------------------------------------------------------------------------

def test_default_filter_constants_are_pinned():
    """The defaults come from the calibration on the released sample.

    Changing them changes the query points of every run, so they are pinned,
    and the functions must use them as their defaults.
    """
    assert CSONIC_test.DEFAULT_MIN_INTENSITY == 15
    assert CSONIC_test.DEFAULT_INTENSITY_WINDOW == 15

    params = inspect.signature(CSONIC_test.filter_dark_queries).parameters
    assert params['min_intensity'].default == CSONIC_test.DEFAULT_MIN_INTENSITY
    assert params['window'].default == CSONIC_test.DEFAULT_INTENSITY_WINDOW
    params = inspect.signature(CSONIC_test.query_points).parameters
    assert params['min_intensity'].default is None
    assert params['window'].default == CSONIC_test.DEFAULT_INTENSITY_WINDOW
    assert params['visibility'].default is None
    assert params['return_counts'].default is False


def test_brightness_keeps_bright_points_and_drops_dark_ones():
    """A point in the bright half stays, a point in the black half goes.

    The two points have their u and v swapped, so a filter that reads the
    image at (u, v) instead of (v, u) gets both of them wrong.
    """
    img = _half_bright()
    coord = np.array([[100.0, 20.0],    # row 20, bright
                      [20.0, 100.0]])   # row 100, black

    keep = CSONIC_test.filter_dark_queries(coord, img)

    assert keep.dtype == bool and keep.shape == (2,)
    assert keep.tolist() == [True, False]


def test_brightness_uses_the_mean_of_the_box():
    """On the edge of the bright half, the box mean decides, not one pixel.

    Rows 57 to 63 of the 15 x 15 box around row 64 are at 60 and the other 8
    rows are black, so the mean is 60 * 7 / 15 = 28 exactly.
    """
    img = _half_bright()
    coord = np.array([[64.0, 64.0]])

    at_28 = CSONIC_test.filter_dark_queries(coord, img, min_intensity=28)
    above_28 = CSONIC_test.filter_dark_queries(coord, img, min_intensity=28.01)
    assert at_28.tolist() == [True]
    assert above_28.tolist() == [False]
    # The pixel itself is black, so a one-pixel box drops it at any threshold.
    assert CSONIC_test.filter_dark_queries(coord, img, min_intensity=1,
                                           window=1).tolist() == [False]


def test_brightness_one_bright_pixel_needs_a_bright_neighbourhood():
    """A single speck of noise in the dark is not a return.

    With the default box its mean is 255 / 225, about 1.1, so the point goes.
    With a one-pixel box it is 255, so the point stays. The coordinates are not
    whole pixels, so this also checks that they are rounded, not truncated.
    """
    img = _half_bright()
    img[100, 30] = 255
    coord = np.array([[29.6, 99.5]])   # rounds to column 30, row 100

    assert CSONIC_test.filter_dark_queries(coord, img, window=15).tolist() == [False]
    assert CSONIC_test.filter_dark_queries(coord, img, window=1).tolist() == [True]
    # Truncation would read column 29, row 99, which is black.
    assert CSONIC_test.filter_dark_queries(np.array([[29.4, 99.4]]), img,
                                           window=1).tolist() == [False]


def test_brightness_threshold_is_inclusive():
    """A box mean of exactly min_intensity is kept; anything below is dropped."""
    coord = np.array([[25.0, 25.0], [0.0, 0.0], [49.0, 49.0]])
    at_15 = np.full((50, 50), 15, dtype=np.uint8)
    at_14 = np.full((50, 50), 14, dtype=np.uint8)

    assert CSONIC_test.filter_dark_queries(coord, at_15, min_intensity=15).all()
    assert not CSONIC_test.filter_dark_queries(coord, at_15, min_intensity=15.001).any()
    assert not CSONIC_test.filter_dark_queries(coord, at_14, min_intensity=15).any()


@pytest.mark.parametrize('min_intensity', [0, 0.0, -5])
def test_brightness_zero_or_less_keeps_everything(min_intensity):
    """A threshold of 0 or less turns the filter off, even on a black image."""
    img = np.zeros((64, 64), dtype=np.uint8)
    coord = np.array([[10.0, 10.0], [63.0, 63.0], [-4.0, 70.0]])

    keep = CSONIC_test.filter_dark_queries(coord, img, min_intensity=min_intensity)

    assert keep.dtype == bool and keep.tolist() == [True, True, True]


def test_brightness_points_on_and_outside_the_border_are_clipped():
    """A point on the border, or just outside it, reads the nearest pixel."""
    img = _half_bright()
    coord = np.array([
        [0.0, 0.0],        # top left corner, bright
        [127.0, 0.0],      # top right corner, bright
        [-5.0, -5.0],      # outside the top left corner: reads (0, 0)
        [200.0, 10.0],     # right of the image: reads column 127, row 10
        [-3.0, 500.0],     # below the image: reads (0, 127), black
        [127.6, 127.6],    # rounds to 128, clipped to 127: black
    ])

    keep = CSONIC_test.filter_dark_queries(coord, img)

    assert keep.tolist() == [True, True, True, True, False, False]


def test_brightness_empty_coord_gives_an_empty_mask():
    img = _half_bright()

    for empty in (np.zeros((0, 2)), [], np.zeros(0)):
        keep = CSONIC_test.filter_dark_queries(empty, img)
        assert keep.dtype == bool and keep.shape == (0,)


@pytest.mark.parametrize('min_intensity', [float('nan'), float('inf'), float('-inf')])
def test_brightness_rejects_a_threshold_that_is_not_finite(min_intensity):
    """A nan threshold would drop every point while the report shows no dark count."""
    with pytest.raises(ValueError, match='min_intensity'):
        CSONIC_test.filter_dark_queries(np.array([[5.0, 5.0]]), _half_bright(),
                                        min_intensity=min_intensity)


@pytest.mark.parametrize('window', [0, -3, 4, 2.5])
def test_brightness_rejects_a_window_with_no_centre(window):
    """The box is centred on the point, so its side must be a positive odd number."""
    with pytest.raises(ValueError, match='window'):
        CSONIC_test.filter_dark_queries(np.array([[5.0, 5.0]]), _half_bright(),
                                        window=window)


# --------------------------------------------------------------------------
# the view filter
# --------------------------------------------------------------------------

def _grid(meta, step=37):
    """Pixels over the whole image, the four corners included."""
    w, h = meta['width'], meta['height']
    us = np.unique(np.r_[np.arange(0, w, step), w - 1]).astype(float)
    vs = np.unique(np.r_[np.arange(0, h, step), h - 1]).astype(float)
    uu, vv = np.meshgrid(us, vs)
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


def _border(meta):
    """Every pixel on the four edges of the image."""
    w, h = meta['width'], meta['height']
    cols = np.arange(w, dtype=float)
    rows = np.arange(h, dtype=float)
    return np.concatenate([np.c_[cols, np.zeros(w)], np.c_[cols, np.full(w, h - 1.0)],
                           np.c_[np.zeros(h), rows], np.c_[np.full(h, w - 1.0), rows]])


# Poses far from the origin: the round trip through them leaves rounding
# errors of about 1e-14, enough to push an edge pixel out of an exact view.
FAR_POSES = [
    _world_pose(),
    _pose(_rot_z(33.0) @ _rot_x(-12.0), (41.3, -17.9, 5.25)),
    _pose(_rot_y(-25.0) @ _rot_z(145.0), (-120.7, 88.1, -30.4)),
]


@pytest.mark.parametrize('meta', [META_L, META_H], ids=['low', 'high'])
@pytest.mark.parametrize('pose_index', range(len(FAR_POSES)))
def test_visible_with_the_same_pose_and_metadata_is_everything(meta, pose_index):
    """Every pixel of image 1 is in view when sonar 2 is the same sonar.

    The top row sits at r_max and column 0 at +azi/2, exactly on the edge of
    the view, so this also checks that an edge point counts as inside,
    whatever the rounding. Two of the poses push more than 500 border pixels
    out by about 1e-14 when the comparison has no tolerance.
    """
    pose = FAR_POSES[pose_index]
    coord = np.concatenate([_grid(meta), _border(meta)])

    keep = CSONIC_test.visible_in_image2(coord, pose, pose, meta, meta)

    assert keep.dtype == bool and keep.shape == (len(coord),)
    assert keep.all()


def test_visible_when_sonar_2_looks_the_other_way_is_nothing():
    """Sonar 2 turned 180 degrees about its z axis sees none of it."""
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_z(180.0))
    coord = _grid(META_L)

    keep = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_L)

    assert not keep.any()


def test_visible_drops_the_bearings_a_narrower_sonar_does_not_cover():
    """Image 1 covers 130 degrees, image 2 only 60: +-30 is the limit.

    The same sensor position, only the azimuth differs, so the bearing alone
    decides.
    """
    pose = _world_pose()
    meta2 = dict(META_L, azi=60)
    inside = [0.0, 10.0, -10.0, 25.0, -25.0]
    outside = [40.0, -40.0, 55.0, -55.0, 64.0, -64.0]
    coord = _pix(inside + outside, 5.0, META_L)

    keep = CSONIC_test.visible_in_image2(coord, pose, pose, META_L, meta2)

    assert keep.tolist() == [True] * len(inside) + [False] * len(outside)


def test_visible_drops_the_ranges_sonar_2_does_not_cover():
    """Sonar 2 sees from 2 m to 5 m, so 1 m and 8 m are out of its view."""
    pose = _world_pose()
    meta2 = dict(META_L, r_min=2.0, r_max=5.0)
    coord = _pix([0.0, 20.0, 0.0, -20.0, 0.0], [3.0, 4.5, 8.0, 9.5, 1.0], META_L)

    keep = CSONIC_test.visible_in_image2(coord, pose, pose, META_L, meta2)

    assert keep.tolist() == [True, True, False, False, False]


@pytest.mark.parametrize('pitch', [15.0, -15.0])
def test_visible_keeps_a_point_whose_arc_is_partly_in_view(pitch):
    """Sonar 2 pitched by 15 degrees: part of a +-10 degree arc stays inside.

    The arc of the centre point spans elevations of 5 to 25 degrees (or -25 to
    -5) in sonar 2, which sees +-10 degrees. The centre of the arc alone is out
    of view, so a filter that ignores the arc drops this point.
    """
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_y(pitch))
    coord = _pix([0.0], 5.0, META_L)

    keep = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_L)

    assert keep.tolist() == [True]


@pytest.mark.parametrize('pitch', [30.0, -30.0])
def test_visible_drops_a_point_whose_arc_is_out_of_view(pitch):
    """Pitched by 30 degrees, the whole arc sits at 20 to 40 degrees: out of view.

    The bearing is still 0 and the range still 5 m, so only the elevation
    check can drop this point.
    """
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_y(pitch))
    coord = _pix([0.0], 5.0, META_L)

    keep = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_L)

    assert keep.tolist() == [False]


def test_visible_samples_the_arc_of_image_1():
    """The arc spans the elevation of image 1, not of image 2.

    With a 4 degree arc and a 15 degree pitch the arc sits at 13 to 17 degrees,
    out of the +-10 degree view of sonar 2. With a 20 degree arc it reaches in.
    """
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_y(15.0))
    coord = _pix([0.0], 5.0, META_L)
    narrow = dict(META_L, elev=4)

    narrow_arc = CSONIC_test.visible_in_image2(coord, pose1, pose2, narrow, META_L)
    wide_arc = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_L)
    assert narrow_arc.tolist() == [False]
    assert wide_arc.tolist() == [True]


def test_visible_uses_the_pose_direction_of_project_keypoints():
    """A yaw with a sideways offset tells the two directions of the pose apart.

    Sonar 2 sits 3 m along +y of sonar 1 and is turned 40 degrees towards the
    points. Swapping the poses turns it away from them.
    """
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_z(-40.0), (0.0, 3.0, 0.0))
    coord = _pix([-20.0, -40.0], [6.0, 7.0], META_L)

    keep = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_H)
    projected = CSONIC_test.project_keypoints(coord, pose1, pose2, META_L, META_H)
    swapped = CSONIC_test.visible_in_image2(coord, pose2, pose1, META_L, META_H)

    # The projection lands inside image 2, so the point must be visible.
    assert np.all((projected[:, 0] > 20) & (projected[:, 0] < 492))
    assert np.all((projected[:, 1] > 20) & (projected[:, 1] < 492))
    assert keep.tolist() == [True, True]
    assert swapped.tolist() == [False, False]


def test_visible_agrees_with_project_keypoints_for_small_motions():
    """A point that projects well inside image 2 is visible.

    The poses move a little in every direction, as consecutive frames do. The
    test asks for enough points inside image 2 that it cannot pass on none.
    """
    rng = np.random.default_rng(0)
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_z(6.0) @ _rot_y(3.0) @ _rot_x(2.0), (0.4, -0.3, 0.1))
    coord = np.column_stack([rng.uniform(0, 511, 400), rng.uniform(0, 511, 400)])

    projected = CSONIC_test.project_keypoints(coord, pose1, pose2, META_L, META_H)
    keep = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_H)

    well_inside = ((projected[:, 0] > 30) & (projected[:, 0] < 482)
                   & (projected[:, 1] > 30) & (projected[:, 1] < 482))
    assert well_inside.sum() >= 50
    assert keep[well_inside].all()
    # 130 degrees against 60: a good part of image 1 must be out of view.
    assert (~keep).sum() >= 100


def test_visible_matches_project_keypoints_exactly_for_a_pure_yaw():
    """Under a pure yaw the whole arc turns with its centre.

    Then a point is visible exactly when its projection lands inside image 2,
    so the two functions must agree on every point away from the edges.
    """
    rng = np.random.default_rng(1)
    pose1 = _world_pose()
    pose2 = _moved(pose1, _rot_z(40.0))
    coord = np.column_stack([rng.uniform(0, 511, 400), rng.uniform(0, 511, 400)])

    projected = CSONIC_test.project_keypoints(coord, pose1, pose2, META_L, META_L)
    keep = CSONIC_test.visible_in_image2(coord, pose1, pose2, META_L, META_L)

    u = projected[:, 0]
    clear = np.abs(u) > 2
    clear &= np.abs(u - META_L['width']) > 2
    inside = (u > 0) & (u < META_L['width'])
    assert inside[clear].sum() >= 50 and (~inside[clear]).sum() >= 50
    assert np.array_equal(keep[clear], inside[clear])


def test_visible_empty_coord_gives_an_empty_mask():
    pose = _world_pose()

    keep = CSONIC_test.visible_in_image2(np.zeros((0, 2)), pose, pose, META_L, META_H)

    assert keep.dtype == bool and keep.shape == (0,)


# --------------------------------------------------------------------------
# query_points
# --------------------------------------------------------------------------

# The fixed detections of the query_points tests, on a 512 x 512 image whose
# rows 0 to 299 are bright and rows 300 to 511 black, with the ROI u >= 100
# and sonar 2 a 60 degree sonar at the same pose. Each point exercises one
# step of the chain; B, F and E sit where two steps would drop them, and the
# first step in the chain must be the one that counts them.
BRIGHT_ROWS = 300
ROI = (100, 511, 0, 511)
POINTS = {
    'A': (256.0, 100.0),   # in the ROI, bright, bearing 0         -> kept
    'B': (50.0, 100.0),    # outside the ROI, bright, bearing 52   -> roi
    'F': (50.0, 400.0),    # outside the ROI, black, bearing 52    -> roi
    'C': (256.0, 400.0),   # in the ROI, black, bearing 0          -> dark
    'E': (453.0, 400.0),   # in the ROI, black, bearing -50        -> dark
    'D': (453.0, 100.0),   # in the ROI, bright, bearing -50       -> not visible
}


def _query_image():
    img = np.zeros((512, 512), dtype=np.uint8)
    img[:BRIGHT_ROWS] = 60
    return img


@pytest.fixture
def fake_detector(monkeypatch):
    """Replace the detector with one that returns a fixed set of points."""
    calls = []
    state = {'coord': np.array(list(POINTS.values()))}

    def fake(img, h, w, superpoint_weights=None):
        calls.append((img.shape, h, w, superpoint_weights))
        coord = np.array(state['coord'], dtype=float).reshape(-1, 2)
        return coord, coord, np.zeros((0, 2)), None, None

    monkeypatch.setattr(detection, 'generate_query_kpts', fake)
    state['calls'] = calls
    return state


def _visibility():
    pose = _world_pose()
    return (pose, pose.copy(), dict(META_L, azi=60))


def test_query_points_counts_each_step_in_order(fake_detector):
    """ROI, then brightness, then view: each point is counted by the first step
    that drops it, and the counts add up."""
    coord, counts = CSONIC_test.query_points(
        _query_image(), META_L, kpt_roi=ROI, visibility=_visibility(),
        return_counts=True)

    assert fake_detector['calls'], 'the detector was never called'
    assert counts == {'detected': 6, 'roi': 2, 'dark': 2, 'not_visible': 1, 'kept': 1}
    assert counts['detected'] - counts['roi'] - counts['dark'] - counts['not_visible'] \
        == counts['kept'] == len(coord)
    assert np.array_equal(coord, np.array([POINTS['A']]))


def test_query_points_without_filters_keeps_every_point(fake_detector):
    """min_intensity=0 and no ROI or poses: the detections pass unchanged."""
    coord, counts = CSONIC_test.query_points(_query_image(), META_L, min_intensity=0,
                                             return_counts=True)

    assert counts == {'detected': 6, 'roi': 0, 'dark': 0, 'not_visible': 0, 'kept': 6}
    assert np.array_equal(coord, np.array(list(POINTS.values())))


def test_query_points_returns_only_the_array_by_default(fake_detector):
    """Existing callers get an (N, 2) array, as before the filters existed."""
    coord = CSONIC_test.query_points(_query_image(), META_L)

    assert isinstance(coord, np.ndarray) and coord.shape == (3, 2)
    # The default brightness filter dropped the three black points, F, C and
    # E, and kept the others in the order they were detected.
    assert np.array_equal(coord, np.array([POINTS['A'], POINTS['B'], POINTS['D']]))


def test_query_points_none_means_the_default_threshold(fake_detector):
    """min_intensity=None is 15: a box mean of 15 stays and a mean of 14 goes."""
    img = np.zeros((512, 512), dtype=np.uint8)
    img[:, :256] = 15
    img[:, 256:] = 14
    fake_detector['coord'] = np.array([[100.0, 200.0], [400.0, 200.0]])

    none_coord, none_counts = CSONIC_test.query_points(img, META_L, min_intensity=None,
                                                       return_counts=True)
    zero_coord = CSONIC_test.query_points(img, META_L, min_intensity=0)

    assert none_counts['dark'] == 1
    assert np.array_equal(none_coord, np.array([[100.0, 200.0]]))
    assert len(zero_coord) == 2


def test_query_points_passes_the_window_on(fake_detector):
    """A one-pixel box keeps a single bright pixel that the default box drops."""
    img = np.zeros((512, 512), dtype=np.uint8)
    img[200, 100] = 255
    fake_detector['coord'] = np.array([[100.0, 200.0]])

    assert len(CSONIC_test.query_points(img, META_L)) == 0
    assert len(CSONIC_test.query_points(img, META_L, window=1)) == 1


def test_query_points_visibility_needs_meta1(fake_detector):
    """Without meta1 there is no geometry for image 1. Fail before detecting."""
    with pytest.raises(ValueError, match='meta1'):
        CSONIC_test.query_points(_query_image(), None, visibility=_visibility())

    assert fake_detector['calls'] == []


def test_query_points_visibility_needs_both_poses(fake_detector):
    pose, _, meta2 = _visibility()

    with pytest.raises(ValueError, match='pose'):
        CSONIC_test.query_points(_query_image(), META_L, visibility=(pose, None, meta2))

    assert fake_detector['calls'] == []


# --------------------------------------------------------------------------
# the default threshold of expectation_matching and of the command line
# --------------------------------------------------------------------------

@pytest.fixture
def recorded(monkeypatch):
    """Replace every heavy step of a run, and record what query_points gets."""
    seen = {'query_points': [], 'load_model': [], 'data': []}
    im = np.zeros((512, 512), dtype=np.uint8)
    pose1, pose2 = _world_pose(), _moved(_world_pose(), None, (0.3, 0.0, 0.0))

    def fake_load_data(img1, img2, pose1_path=None, pose2_path=None,
                       meta1_path=None, meta2_path=None):
        data = {'im1': im, 'im2': im.copy(),
                'pose1': pose1 if pose1_path is not None else None,
                'pose2': pose2 if pose2_path is not None else None,
                'meta1': dict(META_L), 'meta2': dict(META_H)}
        seen['data'].append(data)
        return data

    def fake_query_points(im1, meta1=None, **kwargs):
        seen['query_points'].append(kwargs)
        coord = np.zeros((0, 2))
        if kwargs.get('return_counts'):
            counts = {'detected': 9, 'roi': 1, 'dark': 3, 'not_visible': 2, 'kept': 3}
            return coord, counts
        return coord

    def fake_match_points(model, im1, im2, coord1, **kwargs):
        empty = np.zeros((0, 2))
        return empty, empty.copy(), np.zeros(0)

    def fake_load_model(*args, **kwargs):
        seen['load_model'].append(args)
        return object()

    monkeypatch.setattr(detection, 'load_data', fake_load_data)
    monkeypatch.setattr(CSONIC_test, 'query_points', fake_query_points)
    monkeypatch.setattr(CSONIC_test, 'match_points', fake_match_points)
    monkeypatch.setattr(CSONIC_test, 'load_model', fake_load_model)
    monkeypatch.setattr(CSONIC_test, 'make_matching_figure', lambda *a, **k: None)
    seen['poses'] = (pose1, pose2)
    return seen


def _only_call(seen):
    assert len(seen['query_points']) == 1, 'query_points ran {} times'.format(
        len(seen['query_points']))
    return seen['query_points'][0]


@pytest.mark.parametrize('normalize_input, given, expected', [
    (False, None, 0),       # the thesis pipeline: no brightness filter
    (True, None, 15),       # the default pipeline: the calibrated threshold
    (False, 7, 7),          # an explicit value always wins
    (True, 0, 0),
    (True, 22.5, 22.5),
])
def test_expectation_matching_resolves_the_threshold(recorded, normalize_input,
                                                     given, expected):
    CSONIC_test.expectation_matching('a.npy', 'b.npy', 'a_meta.npy', 'b_meta.npy',
                                     model=object(), normalize_input=normalize_input,
                                     min_intensity=given)

    call = _only_call(recorded)
    assert call['min_intensity'] == expected
    assert call['visibility'] is None


def test_expectation_matching_builds_the_visibility_from_the_poses(recorded):
    """poses=(pose1, pose2); the metadata of image 2 comes from meta2_path."""
    pose1, pose2 = _world_pose(), _moved(_world_pose(), _rot_z(5.0))

    CSONIC_test.expectation_matching('a.npy', 'b.npy', 'a_meta.npy', 'b_meta.npy',
                                     model=object(), poses=(pose1, pose2))

    got1, got2, meta2 = _only_call(recorded)['visibility']
    assert got1 is pose1 and got2 is pose2
    assert meta2 is recorded['data'][-1]['meta2']
    assert meta2 == META_H


@pytest.mark.parametrize('which', [0, 1])
def test_expectation_matching_poses_must_hold_two_poses(recorded, which):
    """A None pose is an error, raised before the model or the data loads."""
    poses = [_world_pose(), _world_pose()]
    poses[which] = None

    with pytest.raises(ValueError, match='poses'):
        CSONIC_test.expectation_matching('a.npy', 'b.npy', 'a_meta.npy', 'b_meta.npy',
                                         model_path='ckpt.pth', poses=tuple(poses))

    assert recorded['load_model'] == []
    assert recorded['data'] == []
    assert recorded['query_points'] == []


def test_expectation_matching_keeps_its_old_positional_order():
    """The new arguments come last, so an old positional call still works."""
    names = list(inspect.signature(CSONIC_test.expectation_matching).parameters)

    assert names == ['img1_path', 'img2_path', 'meta1_path', 'meta2_path', 'model',
                     'model_path', 'th', 'kpt_roi', 'normalize_input',
                     'superpoint_weights', 'bn_mode', 'min_intensity', 'poses']
    params = inspect.signature(CSONIC_test.expectation_matching).parameters
    assert params['min_intensity'].default is None
    assert params['poses'].default is None


BASE_ARGV = ['--img1', 'a.npy', '--img2', 'b.npy', '--meta1', 'a_meta.npy',
             '--meta2', 'b_meta.npy', '--model', 'ckpt.pth']
POSE_ARGV = ['--pose1', 'a_pose.npy', '--pose2', 'b_pose.npy']


@pytest.mark.parametrize('extra, expected', [
    ([], 15),
    (['--legacy-input'], 0),
    (['--legacy-input', '--min-intensity', '7'], 7),
    (['--min-intensity', '0'], 0),
    (['--min-intensity', '30'], 30),
])
def test_main_resolves_the_threshold(recorded, tmp_path, extra, expected):
    CSONIC_test.main(BASE_ARGV + extra + ['--out', str(tmp_path / 'm.png')])

    call = _only_call(recorded)
    assert call['min_intensity'] == expected
    assert call['window'] == CSONIC_test.DEFAULT_INTENSITY_WINDOW
    assert call['visibility'] is None
    assert call['return_counts'] is True


def test_main_passes_the_window_and_the_poses(recorded, tmp_path):
    CSONIC_test.main(BASE_ARGV + POSE_ARGV + [
        '--visible-only', '--intensity-window', '5', '--out', str(tmp_path / 'm.png')])

    call = _only_call(recorded)
    assert call['window'] == 5
    pose1, pose2, meta2 = call['visibility']
    assert pose1 is recorded['poses'][0] and pose2 is recorded['poses'][1]
    assert meta2 == META_H


@pytest.mark.parametrize('poses', [[], ['--pose1', 'a_pose.npy'],
                                   ['--pose2', 'b_pose.npy']])
def test_main_visible_only_needs_both_poses(recorded, tmp_path, poses):
    """The same kind of error as --ransac gives, raised before the model loads."""
    with pytest.raises(ValueError, match='--visible-only needs --pose1 and --pose2'):
        CSONIC_test.main(BASE_ARGV + poses + ['--visible-only',
                                              '--out', str(tmp_path / 'm.png')])

    assert recorded['load_model'] == []
    assert recorded['query_points'] == []


def test_parse_args_filter_defaults():
    args = CSONIC_test.parse_args(BASE_ARGV)

    assert args.min_intensity is None
    assert args.intensity_window == CSONIC_test.DEFAULT_INTENSITY_WINDOW
    assert args.visible_only is False

    args = CSONIC_test.parse_args(BASE_ARGV + ['--min-intensity', '12.5',
                                               '--intensity-window', '9',
                                               '--visible-only'])
    assert args.min_intensity == 12.5 and args.intensity_window == 9
    assert args.visible_only is True


def test_main_prints_the_query_point_counts(monkeypatch, fake_detector, tmp_path, capsys):
    """The real query_points, fed the fixed detections, behind the command line."""
    img = _query_image()
    pose = _world_pose()

    monkeypatch.setattr(detection, 'load_data', lambda *a, **k: {
        'im1': img, 'im2': img.copy(), 'pose1': pose, 'pose2': pose.copy(),
        'meta1': dict(META_L), 'meta2': dict(META_L, azi=60)})
    monkeypatch.setattr(CSONIC_test, 'load_model', lambda *a, **k: object())
    monkeypatch.setattr(CSONIC_test, 'match_points', lambda *a, **k: (
        np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)))
    monkeypatch.setattr(CSONIC_test, 'make_matching_figure', lambda *a, **k: None)

    CSONIC_test.main(BASE_ARGV + POSE_ARGV + ['--visible-only',
                                              '--out', str(tmp_path / 'm.png')])
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith('query points:')]
    # F, C and E are black; of A, B and D only A is inside the 60 degree view.
    assert lines == ['query points: 6 detected, 3 dark, 2 not visible in image 2, 1 kept']

    CSONIC_test.main(BASE_ARGV + ['--roi'] + [str(x) for x in ROI]
                     + ['--min-intensity', '0', '--out', str(tmp_path / 'm.png')])
    lines = [line for line in capsys.readouterr().out.splitlines()
             if line.startswith('query points:')]
    assert lines == ['query points: 6 detected, 2 outside --roi, 4 kept']
