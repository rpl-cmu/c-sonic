"""Tests for CtoFCriterion: the pixel/polar geometry, the two losses, the
validity masks, and equivalence with the pre-cleanup implementation.

Fixtures are local to this module so the file stands on its own.

Equivalence is pinned two ways. The checked-in golden fixture always runs. The
direct comparison against the pre-cleanup module needs that commit in the local
history, so it skips in the public repository, where history starts fresh.
"""
import importlib.util
import math
import os
import subprocess

import numpy as np
import pytest
import torch
from torch.testing import assert_close
from torch.utils.data._utils.collate import default_collate

import config
from conftest import add_legacy_args
from CSONIC.criterion import CtoFCriterion

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_COMMIT = '95e6262'
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
GOLDEN_PATH = os.path.join(FIXTURE_DIR, 'criterion_golden.npz')


def load_golden_builder():
    """Import the script that built the golden fixture, to reuse its inputs."""
    path = os.path.join(FIXTURE_DIR, 'make_criterion_golden.py')
    spec = importlib.util.spec_from_file_location('make_criterion_golden', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


golden_builder = load_golden_builder()


def make_collated_meta(B=2, width=512, height=512, r_min=0.1, r_max=10, elev=12, azi=60):
    """Collate the metadata dict the dataloader really produces.

    The dict holds python ints and one float, so the collated tensors are an
    int64/float64 mix. The criterion must keep that mix rather than flatten it
    to the dtype of the coordinates.
    """
    meta = {'width': width, 'height': height, 'r_min': r_min,
            'r_max': r_max, 'elev': elev, 'azi': azi}
    collated = default_collate([meta] * B)
    return {k: v.reshape(-1, 1) for k, v in collated.items()}


def make_args():
    """Build a small argument namespace for these tests.

    ``add_legacy_args`` puts back the arguments the cleanup removed, so the
    baseline criterion from commit 95e6262 can still be built.
    """
    args = config.get_args([])
    args.num_pts = 50
    args.num_samples = 100
    args.batch_size = 2
    return add_legacy_args(args)


def make_meta(B, width=512, height=512, r_min=0.1, r_max=10.0, elev=12.0, azi=60.0):
    """Build the sonar metadata dict that CSONICModel.set_input produces."""
    values = {'width': width, 'height': height, 'r_min': r_min, 'r_max': r_max,
              'elev': elev, 'azi': azi}
    return {k: torch.full((B, 1), float(v), dtype=torch.float32)
            for k, v in values.items()}


def make_pair(B=2, n=50, seed=0):
    """Build a deterministic batch of query points, poses, image and metadata."""
    torch.manual_seed(seed)
    coord1 = torch.rand(B, n, 2) * torch.tensor([511., 400.])
    T1 = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    T2 = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    T2[:, 0, 3] = 0.3
    T2[:, 1, 3] = 0.1
    T2[:, 2, 3] = 0.0
    im2 = torch.randn(B, 1, 512, 512)
    return {'coord1': coord1, 'T1': T1, 'T2': T2, 'im2': im2,
            'meta1': make_meta(B), 'meta2': make_meta(B, r_max=7.0)}


def make_lit_image(B, height=512, width=512, shadow_row=450):
    """Return a bright image with a dark shadow band, so both masks bite."""
    im = torch.ones(B, 1, height, width)
    im[:, :, shadow_row:, :] = 0.0
    return im


def load_baseline(tmp_path):
    """Import the criterion as it was at the baseline commit.

    Skips when that commit is not in the local history. The public repository
    starts from a single fresh commit, where the golden fixture is the only
    equivalence check that can run.
    """
    result = subprocess.run(['git', 'show', f'{BASELINE_COMMIT}:CSONIC/criterion.py'],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(
            f'commit {BASELINE_COMMIT} is not in this repository, so the pre-cleanup '
            f'criterion cannot be read; tests/fixtures/criterion_golden.npz pins the '
            f'same numbers. git said: {result.stderr.strip()}')
    src = result.stdout
    path = tmp_path / 'baseline_criterion.py'
    path.write_text(src)
    spec = importlib.util.spec_from_file_location('baseline_criterion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# pixel <-> polar geometry
# --------------------------------------------------------------------------

def test_pix_polar_roundtrip_batched_meta():
    """polar_to_pix undoes pix_to_polar when the metadata is a (B, 1) tensor."""
    data = make_pair()
    coord1, meta1 = data['coord1'], data['meta1']
    bearing, rng = CtoFCriterion.pix_to_polar(
        coord1, range_max=meta1['r_max'], range_min=meta1['r_min'],
        image_width=meta1['width'], image_height=meta1['height'],
        bearing_max=meta1['azi'])
    u, v = CtoFCriterion.polar_to_pix(
        (bearing.unsqueeze(2), rng.unsqueeze(2)),
        image_width=meta1['width'], image_height=meta1['height'],
        bearing_max=meta1['azi'], range_max=meta1['r_max'], range_min=meta1['r_min'])
    recovered = torch.stack((u[:, :, 0], v[:, :, 0]), dim=-1)
    assert recovered.shape == coord1.shape
    assert_close(recovered, coord1, atol=1e-3, rtol=0)


def test_pix_polar_accepts_python_floats():
    """Python floats and (B, 1) tensors holding the same values agree.

    Regression for the TypeError raised by torch.deg2rad on a python float.
    """
    data = make_pair()
    coord1, meta1 = data['coord1'], data['meta1']
    floats = dict(range_max=10.0, range_min=0.1, image_width=512.0,
                  image_height=512.0, bearing_max=60.0)
    tensors = dict(range_max=meta1['r_max'], range_min=meta1['r_min'],
                   image_width=meta1['width'], image_height=meta1['height'],
                   bearing_max=meta1['azi'])

    bearing_f, range_f = CtoFCriterion.pix_to_polar(coord1, **floats)
    bearing_t, range_t = CtoFCriterion.pix_to_polar(coord1, **tensors)
    assert bearing_f.shape == coord1.shape[:2]
    assert_close(bearing_f, bearing_t)
    assert_close(range_f, range_t)

    point = (bearing_t.unsqueeze(2), range_t.unsqueeze(2))
    u_f, v_f = CtoFCriterion.polar_to_pix(
        point, image_width=512.0, image_height=512.0, bearing_max=60.0,
        range_max=10.0, range_min=0.1)
    u_t, v_t = CtoFCriterion.polar_to_pix(
        point, image_width=meta1['width'], image_height=meta1['height'],
        bearing_max=meta1['azi'], range_max=meta1['r_max'], range_min=meta1['r_min'])
    assert_close(u_f, u_t)
    assert_close(v_f, v_t)


def test_pix_to_polar_known_points():
    """Corner and centre pixels map to the bearings and ranges we expect."""
    point = torch.tensor([[[0., 0.], [512., 512.], [256., 512.]]])
    bearing, rng = CtoFCriterion.pix_to_polar(
        point, range_max=10.0, range_min=0.1, image_width=512.0,
        image_height=512.0, bearing_max=60.0)
    half_fov = math.radians(60.0) / 2
    assert_close(bearing[0], torch.tensor([half_fov, -half_fov, 0.0]), atol=1e-6, rtol=0)
    assert_close(rng[0], torch.tensor([10.0, 0.1, 0.1]), atol=1e-5, rtol=0)


# --------------------------------------------------------------------------
# cycle consistency loss
# --------------------------------------------------------------------------

def test_cycle_loss_vertical_shift_equals_range_diff_sq():
    """A pure vertical shift costs exactly the squared range difference."""
    criterion = CtoFCriterion(make_args())
    B, n = 2, 101
    p1 = torch.zeros((B, n, 2))
    p1[:, :, 0] = torch.linspace(25, 125, n)
    p1[0, :, 1] = 25.
    p1[1, :, 1] = 125.
    p2 = p1.clone()
    p2[:, :, 1] = p1[:, :, 1] + 100.

    meta1 = make_meta(B)
    loss = criterion.sonar_cycle_consistency_loss(p1, p2, meta1)

    _, range_1 = CtoFCriterion.pix_to_polar(
        p1, range_max=meta1['r_max'], range_min=meta1['r_min'],
        image_width=meta1['width'], image_height=meta1['height'],
        bearing_max=meta1['azi'])
    _, range_2 = CtoFCriterion.pix_to_polar(
        p2, range_max=meta1['r_max'], range_min=meta1['r_min'],
        image_width=meta1['width'], image_height=meta1['height'],
        bearing_max=meta1['azi'])
    expected = (range_1 - range_2) ** 2
    hand_value = (100. * (10.0 - 0.1) / 512.) ** 2  # 1.93359 ** 2
    assert abs(hand_value - 3.73878) < 1e-4
    assert_close(loss.expand_as(expected), expected, atol=1e-4, rtol=0)
    assert abs(loss.item() - hand_value) < 1e-4


def test_cycle_loss_zero_for_identity():
    """A point that loops back onto itself costs nothing."""
    criterion = CtoFCriterion(make_args())
    data = make_pair()
    loss = criterion.sonar_cycle_consistency_loss(
        data['coord1'], data['coord1'].clone(), data['meta1'])
    assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)


# --------------------------------------------------------------------------
# epipolar cost
# --------------------------------------------------------------------------

def test_epipolar_cost_zero_identity_pose():
    """With the same pose and the same point, every arc passes through it."""
    criterion = CtoFCriterion(make_args())
    data = make_pair()
    B, n = data['coord1'].shape[:2]
    im2 = make_lit_image(B)
    cost = criterion.sonar_epipolar_cost(
        data['coord1'], data['coord1'].clone(), data['T1'], data['T1'],
        im2, data['meta1'], data['meta1'])
    assert cost.numel() == B * n
    assert_close(cost, torch.zeros_like(cost), atol=1e-4, rtol=0)


def test_epipolar_cost_zero_at_true_projection():
    """The cost vanishes where the point really projects on the arc."""
    args = make_args()
    args.num_samples = 101  # odd, so phi = 0 is sampled
    criterion = CtoFCriterion(args)
    data = make_pair()
    coord1, meta1, meta2 = data['coord1'], data['meta1'], data['meta2']
    B = coord1.shape[0]
    # A vertical baseline spreads the arc, so the minimum over it is the only
    # sample that costs nothing.
    T2 = data['T2'].clone()
    T2[:, 2, 3] = 1.0

    bearing_1, dist_1 = CtoFCriterion.pix_to_polar(
        coord1, range_max=meta1['r_max'], range_min=meta1['r_min'],
        image_width=meta1['width'], image_height=meta1['height'],
        bearing_max=meta1['azi'])
    phi = criterion.batch_linspace(-meta1['elev'] / 2, meta1['elev'] / 2, args.num_samples)
    mid = args.num_samples // 2
    assert abs(phi[0, mid].item()) < 1e-6
    arc = criterion.convert_to_arc(dist_1, bearing_1, phi)
    rot, t = criterion.get_rel_pose(data['T1'], T2)
    arc_t = criterion.transform_arc_points(arc, rot, t)
    bearing_t, range_t = criterion.cartesian_arc_to_polar(arc_t)
    u, v = CtoFCriterion.polar_to_pix(
        (bearing_t[:, :, mid:mid + 1], range_t[:, :, mid:mid + 1]),
        image_width=meta2['width'], image_height=meta2['height'],
        bearing_max=meta2['azi'], range_max=meta2['r_max'], range_min=meta2['r_min'])
    coord2 = torch.stack((u[:, :, 0], v[:, :, 0]), dim=-1)

    im2 = make_lit_image(B)
    cost = criterion.sonar_epipolar_cost(
        coord1, coord2, data['T1'], T2, im2, meta1, meta2)
    assert cost.numel() > 0, 'every point was masked out; the test would be vacuous'
    assert_close(cost, torch.zeros_like(cost), atol=1e-3, rtol=0)

    # Moving the match five pixels off the arc costs more at every point, so the
    # check above cannot pass on a cost that is zero whatever the input.
    off_arc = criterion.sonar_epipolar_cost(
        coord1, coord2 + 5.0, data['T1'], T2, im2, meta1, meta2)
    assert off_arc.numel() == cost.numel()
    assert off_arc.min().item() > 5e-4


def test_epipolar_cost_positive_under_translation():
    """Moving the sonar a metre makes the same pixel a wrong correspondence."""
    criterion = CtoFCriterion(make_args())
    data = make_pair()
    B = data['coord1'].shape[0]
    T2 = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    T2[:, 0, 3] = 1.0
    cost = criterion.sonar_epipolar_cost(
        data['coord1'], data['coord1'].clone(), data['T1'], T2,
        make_lit_image(B), data['meta1'], data['meta1'])
    assert cost.numel() > 0, 'every point was masked out; the test would be vacuous'
    assert cost.mean().item() > 0.1


# --------------------------------------------------------------------------
# validity masks
# --------------------------------------------------------------------------

def test_mask_invalid_points_drops_out_of_bounds():
    """A single arc sample outside the image drops the whole point."""
    criterion = CtoFCriterion(make_args())
    B, n, S = 1, 5, 4
    euc_dist = torch.arange(float(n)).reshape(B, n)
    u = torch.full((B, n, S), 100.)
    v = torch.full((B, n, S), 100.)
    h = torch.full((B, 1), 512.)
    w = torch.full((B, 1), 512.)
    u[0, 3, 2] = 522.  # w + 10

    im2 = make_lit_image(B)
    valid = criterion.mask_invalid_points(euc_dist, u, v, h, w, im2)
    assert_close(valid, torch.tensor([0., 1., 2., 4.]))

    debug = criterion.mask_invalid_points(euc_dist, u, v, h, w, im2, debug=True)
    assert len(debug) == 5
    assert_close(debug[0], valid)
    assert debug[4].tolist() == [[True, True, True, False, True]]


def test_mask_invalid_points_intensity_drops_shadow():
    """An arc that lands in the shadow band drops the point."""
    criterion = CtoFCriterion(make_args())
    B, n, S = 1, 3, 3
    euc_dist = torch.tensor([[1., 2., 3.]])
    u = torch.full((B, n, S), 100.)
    v = torch.full((B, n, S), 100.)
    v[0, 1, :] = 460.  # whole arc below the shadow row
    v[0, 2, 1] = 460.  # a single sample in shadow still drops the point
    h = torch.full((B, 1), 512.)
    w = torch.full((B, 1), 512.)
    im2 = make_lit_image(B)

    valid = criterion.mask_invalid_points_intensity(euc_dist, u, v, h, w, im2)
    assert_close(valid, torch.tensor([1.]))

    debug = criterion.mask_invalid_points_intensity(euc_dist, u, v, h, w, im2, debug=True)
    assert len(debug) == 6
    assert debug[4].tolist() == [[[True] * S, [False] * S, [True, False, True]]]


# --------------------------------------------------------------------------
# forward
# --------------------------------------------------------------------------

def make_out(coord1, seed=1):
    """Build the network output dict the criterion consumes."""
    torch.manual_seed(seed)
    noise = torch.randn_like(coord1) * 2.0
    out = {}
    for key, scale in (('coord2_ec', 1.0), ('coord2_ef', 0.5),
                       ('coord1_lc', 0.3), ('coord1_lf', 0.1)):
        out[key] = (coord1 + noise * scale).detach().requires_grad_(True)
    out.update({'coarse_h': 32, 'coarse_w': 32, 'fine_h': 128, 'fine_w': 128})
    return out


def test_forward_returns_five_finite_and_differentiable():
    """forward returns five finite terms and back-propagates into the fine match."""
    criterion = CtoFCriterion(make_args())
    data = make_pair()
    B = data['coord1'].shape[0]
    im2 = make_lit_image(B)
    out = make_out(data['coord1'])

    result = criterion(data['coord1'], out, data['T1'], data['T2'], im2,
                       data['meta1'], data['meta2'])
    assert len(result) == 5
    for term in result:
        assert torch.is_tensor(term)
        assert torch.isfinite(term).all()

    loss = result[0]
    loss.backward()
    for key in ('coord2_ec', 'coord2_ef', 'coord1_lc', 'coord1_lf'):
        grad = out[key].grad
        assert grad is not None, f'no gradient reached {key}'
        assert torch.isfinite(grad).all(), f'non-finite gradient on {key}'
        assert grad.abs().sum() > 0, f'zero gradient on {key}'


def test_w_cycle_fine_is_used():
    """The fine cycle weight comes from w_cycle_fine, not w_cycle_coarse.

    Checked end to end: with every other weight zero, the total loss is exactly
    w_cycle_fine times the fine cycle term.
    """
    args = make_args()
    args.w_epipolar_coarse = 0.0
    args.w_epipolar_fine = 0.0
    args.w_cycle_coarse = 1.0
    args.w_cycle_fine = 3.0
    criterion = CtoFCriterion(args)
    assert criterion.w_cf == 3.0
    assert criterion.w_cc == 1.0

    data = make_pair()
    B = data['coord1'].shape[0]
    out = make_out(data['coord1'])
    # Drop the coarse cycle term too, so only the fine one is left.
    criterion.w_cc = 0.0
    loss, _, _, _, closs_f = criterion(
        data['coord1'], out, data['T1'], data['T2'], make_lit_image(B),
        data['meta1'], data['meta2'])
    assert closs_f.abs().sum() > 0, 'a zero fine cycle term would make this vacuous'
    assert_close(loss, torch.mean(3.0 * closs_f))


# --------------------------------------------------------------------------
# equivalence with the pre-cleanup implementation
# --------------------------------------------------------------------------

def _equivalence_inputs():
    """Inputs that leave both masks non-trivial and every loss term finite."""
    torch.manual_seed(0)
    B, n = 2, 50
    coord1 = torch.rand(B, n, 2) * torch.tensor([511., 400.])
    T1 = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    T2 = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    T2[:, 0, 3] = 0.3
    T2[:, 1, 3] = 0.1
    im2 = torch.rand(B, 1, 512, 512) * 0.5 + 0.5
    im2[:, :, 450:, :] = torch.rand(B, 1, 62, 512) * 0.05
    return {'coord1': coord1, 'T1': T1, 'T2': T2, 'im2': im2,
            'meta1': make_meta(B), 'meta2': make_meta(B, r_max=7.0)}


def test_matches_baseline_implementation(tmp_path):
    """The cleaned criterion reproduces the baseline numbers exactly."""
    baseline = load_baseline(tmp_path)
    args = make_args()
    args.w_cycle_fine = args.w_cycle_coarse  # neutralise the fixed typo
    data = _equivalence_inputs()
    out = make_out(data['coord1'])

    old = baseline.CtoFCriterion(args)(
        data['coord1'], out, data['T1'], data['T2'], None, data['im2'],
        data['im2'], data['meta1'], data['meta2'])
    new = CtoFCriterion(args)(
        data['coord1'], out, data['T1'], data['T2'], data['im2'],
        data['meta1'], data['meta2'])

    assert len(old) == 6 and len(new) == 5
    for term in new:
        assert torch.isfinite(term).all(), 'degenerate inputs would make this vacuous'
    assert new[0].abs().item() > 0
    for i in range(5):
        assert_close(new[i], old[i])


def test_baseline_epipolar_cost_raises_name_error(tmp_path):
    """The baseline only defined u, v under debug; the cleaned code always does."""
    baseline = load_baseline(tmp_path)
    data = make_pair()
    old = baseline.CtoFCriterion(make_args())
    with pytest.raises(NameError):
        old.sonar_epipolar_cost(
            data['coord1'], data['coord1'].clone(), data['T1'], data['T2'],
            data['im2'], data['im2'], data['meta1'], data['meta2'], debug=False)


# --------------------------------------------------------------------------
# metadata dtypes
# --------------------------------------------------------------------------

def test_bcast_preserves_tensor_dtype_and_casts_numbers():
    """A metadata tensor keeps its dtype; a python number takes the reference's.

    The collated metadata is an int64/float64 mix. Casting it to the dtype of the
    coordinates would silently round float64 ranges to float32 (0.1 becomes
    0.10000000149) and change the dtype of every result.
    """
    like = torch.zeros(2, 5, dtype=torch.float32)

    cases = [
        (torch.tensor([[0.1], [0.1]], dtype=torch.float64), torch.float64, (2, 1)),
        (torch.tensor([[512], [512]], dtype=torch.int64), torch.int64, (2, 1)),
        (torch.tensor([60, 60], dtype=torch.int64), torch.int64, (2, 1)),
        (torch.tensor(0.1, dtype=torch.float64), torch.float64, ()),
        (512, torch.float32, ()),
        (0.1, torch.float32, ()),
    ]
    for value, expected_dtype, expected_shape in cases:
        result = CtoFCriterion._bcast(value, like)
        assert result.dtype == expected_dtype, f'{value!r} became {result.dtype}'
        assert tuple(result.shape) == expected_shape, f'{value!r} became {tuple(result.shape)}'
        assert (result * like).shape == like.shape, f'{value!r} does not broadcast'

    # float64 precision survives, which is the whole point.
    exact = CtoFCriterion._bcast(torch.tensor([[0.1], [0.1]], dtype=torch.float64), like)
    assert exact[0, 0].item() == 0.1
    assert float(torch.tensor(0.1, dtype=torch.float32)) != 0.1

    # A (B, n, S) reference gets a (B, 1, 1) metadata tensor.
    like3 = torch.zeros(2, 5, 7, dtype=torch.float32)
    wide = CtoFCriterion._bcast(torch.tensor([[7.], [7.]], dtype=torch.float64), like3)
    assert tuple(wide.shape) == (2, 1, 1)


def test_real_metadata_dtypes_match_baseline(tmp_path):
    """Collated int64/float64 metadata gives the pre-cleanup numbers and dtypes."""
    baseline = load_baseline(tmp_path)
    args = make_args()
    args.w_cycle_fine = args.w_cycle_coarse  # neutralise the fixed typo
    old = baseline.CtoFCriterion(args)
    new = CtoFCriterion(args)

    data = make_pair()
    coord1 = data['coord1']
    B = coord1.shape[0]
    im2 = make_lit_image(B)
    meta1 = make_collated_meta(B, r_max=10)
    meta2 = make_collated_meta(B, r_max=7)
    assert meta1['width'].dtype == torch.int64
    assert meta1['r_min'].dtype == torch.float64
    out = make_out(coord1)

    pairs = []
    pairs.append((
        new.sonar_epipolar_cost(coord1, out['coord2_ec'], data['T1'], data['T2'],
                                im2, meta1, meta2),
        old.sonar_epipolar_cost(coord1, out['coord2_ec'], data['T1'], data['T2'],
                                im2, im2, meta1, meta2),
    ))
    pairs.append((
        new.sonar_cycle_consistency_loss(coord1, out['coord1_lc'], meta1),
        old.sonar_cycle_consistency_loss(coord1, out['coord1_lc'], None, im2, im2,
                                         meta1, meta2, "coarse"),
    ))
    new_forward = new(coord1, out, data['T1'], data['T2'], im2, meta1, meta2)
    old_forward = old(coord1, out, data['T1'], data['T2'], None, im2, im2, meta1, meta2)
    pairs.extend(zip(new_forward, old_forward[:5]))

    assert pairs[0][0].numel() > 0, 'every point was masked out; the test would be vacuous'
    for got, want in pairs:
        assert torch.isfinite(got).all(), 'degenerate inputs would make this vacuous'
        assert got.dtype == want.dtype, f'dtype drifted: {got.dtype} vs {want.dtype}'
        assert_close(got, want, rtol=0, atol=0)

    # The float64 r_min really does reach the output, so the check has teeth.
    assert new_forward[0].dtype == torch.float64


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_half_coords_keep_baseline_dtypes_on_cuda(tmp_path):
    """Half-precision coordinates with float32 metadata keep the baseline dtype."""
    baseline = load_baseline(tmp_path)
    device = torch.device('cuda')
    coord = (torch.stack([torch.linspace(10., 500., 8),
                          torch.linspace(60., 400., 8)], dim=-1)
             .unsqueeze(0).repeat(2, 1, 1).to(device=device, dtype=torch.float16))
    meta = {k: v.to(device=device, dtype=torch.float32)
            for k, v in make_meta(2).items()}

    new_b, new_r = CtoFCriterion.pix_to_polar(
        coord, range_max=meta['r_max'], range_min=meta['r_min'],
        image_width=meta['width'], image_height=meta['height'],
        bearing_max=meta['azi'])
    old_b, old_r = baseline.CtoFCriterion.pix_to_polar(
        coord, bearing_center=meta['width'] / 2, range_max=meta['r_max'],
        range_min=meta['r_min'], image_width=meta['width'],
        image_height=meta['height'], bearing_max=meta['azi'])
    assert new_b.dtype == old_b.dtype == torch.float32
    assert new_r.dtype == old_r.dtype == torch.float32
    assert_close(new_b, old_b, rtol=0, atol=0)
    assert_close(new_r, old_r, rtol=0, atol=0)

    point = (new_b.unsqueeze(2), new_r.unsqueeze(2))
    new_u, new_v = CtoFCriterion.polar_to_pix(
        point, image_width=meta['width'], image_height=meta['height'],
        bearing_max=meta['azi'], range_max=meta['r_max'], range_min=meta['r_min'])
    old_u, old_v = baseline.CtoFCriterion.polar_to_pix(
        point, image_width=meta['width'], image_height=meta['height'],
        bearing_max=meta['azi'], range_max=meta['r_max'], range_min=meta['r_min'])
    assert new_u.dtype == old_u.dtype
    assert new_v.dtype == old_v.dtype
    assert_close(new_u, old_u, rtol=0, atol=0)
    assert_close(new_v, old_v, rtol=0, atol=0)


# --------------------------------------------------------------------------
# golden fixture
# --------------------------------------------------------------------------

def test_matches_golden_outputs():
    """The criterion reproduces the checked-in pre-cleanup numbers.

    This runs everywhere, including the public repository, where the pre-cleanup
    commit no longer exists. Regenerate with
    ``tests/fixtures/make_criterion_golden.py``.
    """
    golden = np.load(GOLDEN_PATH)
    assert str(golden['baseline_commit']) == BASELINE_COMMIT
    assert int(golden['num_samples']) == golden_builder.NUM_SAMPLES

    criterion = CtoFCriterion(golden_builder.make_criterion_args())
    data = golden_builder.make_inputs()

    got = {
        'epipolar_cost_ec': criterion.sonar_epipolar_cost(
            data['coord1'], data['out']['coord2_ec'], data['T1'], data['T2'],
            data['im2'], data['meta1'], data['meta2']),
        'cycle_loss_lc': criterion.sonar_cycle_consistency_loss(
            data['coord1'], data['out']['coord1_lc'], data['meta1']),
    }
    forward = criterion(data['coord1'], data['out'], data['T1'], data['T2'],
                        data['im2'], data['meta1'], data['meta2'])
    got.update(zip(('loss', 'eloss_c', 'eloss_f', 'closs_c', 'closs_f'), forward))

    # The fixture must pin a live mask, not an empty one.
    assert got['epipolar_cost_ec'].numel() == golden['epipolar_cost_ec'].shape[0]
    assert 0 < got['epipolar_cost_ec'].numel() < data['coord1'][:, :, 0].numel()

    for key, value in got.items():
        want = torch.as_tensor(golden[key])
        assert value.dtype == want.dtype, f'{key} dtype drifted: {value.dtype} vs {want.dtype}'
        assert_close(value, want)


# --------------------------------------------------------------------------
# shadow row on a smaller image
# --------------------------------------------------------------------------

def test_shadow_row_scales_with_image_height():
    """On a 256-row image the shadow band starts at row 225, not 450."""
    criterion = CtoFCriterion(make_args())
    B, n, S = 1, 3, 3
    height = width = 256
    assert height * 450 // 512 == 225

    im2 = torch.ones(B, 1, height, width)
    im2[:, :, 225:, :] = 0.0
    euc_dist = torch.tensor([[1., 2., 3.]])
    u = torch.full((B, n, S), 100.)
    v = torch.full((B, n, S), 100.)
    v[0, 1, :] = 230.  # inside the scaled shadow band
    v[0, 2, 1] = 224.  # just above it, so the point survives
    h = torch.full((B, 1), float(height))
    w = torch.full((B, 1), float(width))

    valid = criterion.mask_invalid_points_intensity(euc_dist, u, v, h, w, im2)
    assert_close(valid, torch.tensor([1., 3.]))
