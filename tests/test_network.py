"""Tests for CSONICNet: both attention modes, the released checkpoint layout,
and equivalence with the pre-cleanup implementation."""
import copy
import importlib.util
import os
import subprocess
import sys

import numpy
import pytest
import torch

from conftest import add_legacy_args
from CSONIC.network import CSONICNet, ImageCrossAttention
from tests.fixtures.make_network_golden import (GOLDEN_KEYS, GOLDEN_PATH, build_inputs,
                                                build_released_net)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(REPO_ROOT, 'pretrained', 'CSONIC_pretrained.pth')
BASELINE_COMMIT = '95e6262'

BASE_KEYS = {
    'coord2_ec', 'coord2_ef', 'coord1_lc', 'coord1_lf',
    'std_c', 'std_f', 'std_lc', 'std_lf',
    'coarse_h', 'coarse_w', 'fine_h', 'fine_w',
}
COORD_KEYS = ('coord2_ec', 'coord2_ef', 'coord1_lc', 'coord1_lf')
STD_KEYS = ('std_c', 'std_f', 'std_lc', 'std_lf')
SHAPE_KEYS = ('coarse_h', 'coarse_w', 'fine_h', 'fine_w')


def build_net(args, device):
    """Build an untrained network in eval mode."""
    net = CSONICNet(args, device)
    net.eval()
    return net


def check_common_outputs(out, args, n_pts=50, w=512):
    """Check shapes, finiteness and coordinate ranges of a forward pass."""
    for key in COORD_KEYS:
        assert out[key].shape == (2, n_pts, 2), key
        assert torch.isfinite(out[key]).all(), key
    for key in STD_KEYS:
        assert out[key].shape == (2, n_pts), key
        assert torch.isfinite(out[key]).all(), key
    # The coarse predictions average over the whole image grid, so they stay
    # inside the image.
    for key in ('coord2_ec', 'coord1_lc'):
        assert out[key].min() >= 0.0, key
        assert out[key].max() <= w - 1, key
    # The fine predictions average over a local window that may hang over the
    # border by window_size of the normalized range.
    margin = args.window_size * (w - 1) / 2.0 + 1e-3
    for key in ('coord2_ef', 'coord1_lf'):
        assert out[key].min() >= -margin, key
        assert out[key].max() <= w - 1 + margin, key
    assert out['coarse_h'] == 32 and out['coarse_w'] == 32
    assert out['fine_h'] == 128 and out['fine_w'] == 128


def test_forward_cross_attention_keys_shapes_finite(args, device, synthetic_pair):
    args.cross_attention = 1
    net = build_net(args, device)
    assert hasattr(net, 'net_prep') and not hasattr(net, 'net')
    with torch.no_grad():
        out = net(synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1'])
    assert set(out) == BASE_KEYS | {'ac1_weights', 'ac2_weights'}
    check_common_outputs(out, args)
    for key in ('ac1_weights', 'ac2_weights'):
        assert torch.isfinite(out[key]).all(), key
        assert out[key].shape[:2] == (2, 8), key  # batch, heads


def test_forward_baseline_keys_shapes_finite(args, device, synthetic_pair):
    args.cross_attention = 0
    net = build_net(args, device)
    assert hasattr(net, 'net')
    assert not hasattr(net, 'net_prep')
    with torch.no_grad():
        out = net(synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1'])
    assert set(out) == BASE_KEYS
    check_common_outputs(out, args)


@pytest.mark.parametrize('cross_attention', [0, 1])
def test_test_method_both_modes(args, device, synthetic_pair, cross_attention):
    args.cross_attention = cross_attention
    net = build_net(args, device)
    with torch.no_grad():
        coord2_ef, std = net.test(synthetic_pair['im1'], synthetic_pair['im2'],
                                  synthetic_pair['coord1'])
    assert coord2_ef.shape == (2, 50, 2)
    assert std.shape == (2, 50)
    assert torch.isfinite(coord2_ef).all()
    assert torch.isfinite(std).all()


@pytest.mark.parametrize('cross_attention', [0, 1])
def test_extract_features_both_modes(args, device, synthetic_pair, cross_attention):
    args.cross_attention = cross_attention
    net = build_net(args, device)
    im1, im2, coord1 = synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1']
    with torch.no_grad():
        if cross_attention:
            feat_c, feat_f = net.extract_features(im1, coord1, im_ref=im2)
        else:
            feat_c, feat_f = net.extract_features(im1, coord1)
    assert feat_c.shape == (2, 50, args.coarse_feat_dim)
    assert feat_f.shape == (2, 50, args.fine_feat_dim)
    assert torch.isfinite(feat_c).all() and torch.isfinite(feat_f).all()


def test_extract_features_cross_attention_requires_reference(args, device, synthetic_pair):
    args.cross_attention = 1
    net = build_net(args, device)
    with torch.no_grad(), pytest.raises(ValueError):
        net.extract_features(synthetic_pair['im1'], synthetic_pair['coord1'])


def test_accepts_3d_input(args, device, synthetic_pair):
    args.cross_attention = 1
    net = build_net(args, device)
    im1, im2, coord1 = synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1']
    with torch.no_grad():
        out_4d = net(im1, im2, coord1)
        out_3d = net(im1.squeeze(1), im2.squeeze(1), coord1)
    assert im1.squeeze(1).dim() == 3
    for key in COORD_KEYS + STD_KEYS:
        torch.testing.assert_close(out_3d[key], out_4d[key])
    for key in SHAPE_KEYS:
        assert out_3d[key] == out_4d[key]


def test_state_dict_layout_matches_release(args, device):
    args.cross_attention = 1
    net = build_net(args, device)
    keys = list(net.state_dict())
    prefixes = {}
    for key in keys:
        prefixes[key.split('.')[0]] = prefixes.get(key.split('.')[0], 0) + 1
    assert set(prefixes) == {'net_prep', 'net_coarse_fine',
                             'cross_attention_coarse', 'cross_attention_fine'}
    assert prefixes == {'net_prep': 174, 'net_coarse_fine': 42,
                        'cross_attention_coarse': 13, 'cross_attention_fine': 13}
    assert len(keys) == 242


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(CKPT_PATH), reason='released checkpoint not available')
def test_released_checkpoint_loads_strict(args, device, synthetic_pair):
    args.cross_attention = 1
    net = build_net(args, device)
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['state_dict'], strict=True)
    net.to(device)
    net.eval()
    with torch.no_grad():
        out = net(synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1'])
    check_common_outputs(out, args)


def _load_baseline_module(tmp_path):
    """Import CSONIC/network.py as it was before the cleanup.

    The public repository is a single fresh commit, so the baseline is not
    always reachable. Skip instead of failing when it is gone.
    """
    try:
        source = subprocess.check_output(
            ['git', 'show', '%s:CSONIC/network.py' % BASELINE_COMMIT],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip('baseline commit %s not available' % BASELINE_COMMIT)
    path = tmp_path / 'baseline_network.py'
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('baseline_network', str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules['baseline_network'] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.parametrize('device_name', ['cpu', 'accelerator'])
def test_matches_baseline_implementation(args, device, synthetic_pair, tmp_path, device_name):
    """The cleanup must not change what the network computes.

    The two implementations agree bit for bit on every device. The rotary
    frequency table is built on the CPU and only then moved, so results on an
    accelerator still match the paper-era code.
    """
    run_device = torch.device('cpu') if device_name == 'cpu' else device
    if device_name == 'accelerator' and run_device.type == 'cpu':
        pytest.skip('no accelerator, the cpu case already covers this')
    baseline = _load_baseline_module(tmp_path)
    args.cross_attention = 1
    new_net = build_net(args, run_device)
    # The baseline reads options the cleanup removed; put them back on a copy.
    old_net = baseline.CSONICNet(add_legacy_args(copy.deepcopy(args)), run_device)
    old_net.load_state_dict(new_net.state_dict(), strict=True)
    old_net.to(run_device)
    old_net.eval()

    im1 = synthetic_pair['im1'].to(run_device)
    im2 = synthetic_pair['im2'].to(run_device)
    coord1 = synthetic_pair['coord1'].to(run_device)
    meta1 = {k: v.to(run_device) for k, v in synthetic_pair['meta1'].items()}
    meta2 = {k: v.to(run_device) for k, v in synthetic_pair['meta2'].items()}
    with torch.no_grad():
        new_out = new_net(im1, im2, coord1)
        old_out = old_net(im1, im2, coord1, meta1, meta2)

    for key in COORD_KEYS + STD_KEYS + ('ac1_weights', 'ac2_weights'):
        torch.testing.assert_close(new_out[key], old_out[key], rtol=0, atol=0, msg=key)
    for key in SHAPE_KEYS:
        assert new_out[key] == old_out[key], key


@pytest.mark.slow
def test_baseline_crashes_without_cross_attention(args, device, synthetic_pair, tmp_path):
    """The old code never built self.net, so the SONIC baseline could not run."""
    baseline = _load_baseline_module(tmp_path)
    args.cross_attention = 0
    old_net = baseline.CSONICNet(add_legacy_args(copy.deepcopy(args)), device)
    old_net.eval()
    with torch.no_grad(), pytest.raises(AttributeError):
        old_net(synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1'],
                synthetic_pair['meta1'], synthetic_pair['meta2'])
    new_net = build_net(args, device)
    with torch.no_grad():
        out = new_net(synthetic_pair['im1'], synthetic_pair['im2'], synthetic_pair['coord1'])
    assert set(out) == BASE_KEYS


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(CKPT_PATH), reason='released checkpoint not available')
@pytest.mark.skipif(not os.path.exists(GOLDEN_PATH), reason='golden outputs not available')
def test_matches_golden_outputs():
    """The released checkpoint must keep producing the checked-in outputs.

    The inputs come from the same helper the generator script uses, so the two
    cannot drift. Regenerate with tests/fixtures/make_network_golden.py.
    Everything runs on CPU; the tolerance covers convolution kernels that differ
    between machines.
    """
    net = build_released_net('cpu')
    im1, im2, coord1 = build_inputs('cpu')
    with torch.no_grad():
        out = net(im1, im2, coord1)
    golden = numpy.load(GOLDEN_PATH)
    assert set(golden.files) == set(GOLDEN_KEYS)
    for key in GOLDEN_KEYS:
        expected = torch.from_numpy(golden[key])
        torch.testing.assert_close(out[key], expected, rtol=0, atol=1e-3, msg=key)
    # guard against a golden file of constants that any output would match
    assert golden['coord2_ef'].std() > 1.0
    assert golden['std_f'].std() > 0.0


def test_cross_attention_non_divisible_N(device):
    """The chunked attention must handle a token count that the chunks do not divide."""
    torch.manual_seed(0)
    block = ImageCrossAttention(feature_dim=256).to(device).eval()
    x = torch.randn(1, 256, 33, 31, device=device)
    assert (33 * 31) % 8 != 0
    with torch.no_grad():
        out, weights = block(x, x, x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert weights.shape == (1, 8, 33 * 31, 33 * 31)


def test_normalize_denormalize_roundtrip():
    torch.manual_seed(0)
    coord = torch.rand(2, 50, 2) * torch.tensor([511., 399.])
    coord_n = CSONICNet.normalize(coord, 400, 512)
    assert coord_n.min() >= -1.0 and coord_n.max() <= 1.0
    torch.testing.assert_close(CSONICNet.denormalize(coord_n, 400, 512), coord)
