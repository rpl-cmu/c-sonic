"""Shared fixtures for the test suite."""
import copy
import os

import pytest
import torch

import config

# The tiny dataset the slow tests read. Set CSONIC_TINY_DATASET to point the
# suite at a dataset in the README layout; the tests skip when it is unset or
# the directory is missing.
TINY_DATASET_DIR = os.environ.get('CSONIC_TINY_DATASET', '')
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERPOINT_WEIGHTS = os.path.join(REPO_ROOT, 'pretrained', 'superpoint_v1.pth')

# Arguments the cleanup removed from config.py. The pre-cleanup modules from the
# baseline commit still read them, so the equivalence tests put them back on the
# namespace by hand. Nothing in the released code touches these.
LEGACY_ARGS = {
    'attention': 0,
    'range_max': 10.0,
    'range_end': 10.0,
    'range_start': 0.1,
    'horizontal_fov': 130.0,
    'vertical_fov': 20.0,
    'th_epipolar': 1.0,
    'th_cycle': 1.0,
    'w_std': 0.0,
    'std': 1,
    'curriculum': 0,
    'curr_diff': 'all',
    'prune_kp': 0,
    'train_kp': 'mixed',
    'akaze_pts': 5,
}


def add_legacy_args(args):
    """Add the arguments the pre-cleanup modules read, and return args.

    These exist only so the modules from commit 95e6262 can be instantiated for
    the baseline-equivalence tests. The released code reads none of them, and an
    attribute already present is left alone.
    """
    for name, value in LEGACY_ARGS.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def make_args():
    """Build a small argument namespace for tests."""
    args = config.get_args([])
    args.pretrained = 0
    args.cross_attention = 1
    args.num_pts = 50
    args.num_samples = 100
    args.batch_size = 2
    args.backbone = 'resnet34'
    args.coarse_feat_dim = 64
    args.fine_feat_dim = 64
    return args


@pytest.fixture(scope='session')
def _base_args():
    return make_args()


@pytest.fixture
def args(_base_args):
    """A fresh copy of the test arguments. Tests may mutate it.

    Deliberately free of the legacy attributes: a production module that starts
    reading a removed option again must fail here. Only the tests that build the
    pre-cleanup modules call add_legacy_args, on their own copy.
    """
    return copy.deepcopy(_base_args)


@pytest.fixture(scope='session')
def tiny_dataset_dir():
    """Path to the tiny sonar dataset. Skips when it is not available."""
    if not TINY_DATASET_DIR or not os.path.isdir(TINY_DATASET_DIR):
        pytest.skip('set CSONIC_TINY_DATASET to a dataset in the README layout '
                    '(got {!r})'.format(TINY_DATASET_DIR))
    if not os.path.isfile(SUPERPOINT_WEIGHTS):
        pytest.skip('SuperPoint weights not found at {}'.format(SUPERPOINT_WEIGHTS))
    return TINY_DATASET_DIR


@pytest.fixture(scope='session')
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _make_meta(B, width=512, height=512, r_min=0.1, r_max=10.0, elev=12.0,
               azi=60.0, device='cpu'):
    """Build the sonar metadata dict that CSONICModel.set_input produces."""
    values = {'width': width, 'height': height, 'r_min': r_min, 'r_max': r_max,
              'elev': elev, 'azi': azi}
    return {k: torch.full((B, 1), float(v), dtype=torch.float32, device=device)
            for k, v in values.items()}


@pytest.fixture
def make_meta():
    """Return the metadata builder so tests can pick their own values."""
    return _make_meta


@pytest.fixture
def synthetic_pair(device):
    """A deterministic image pair with query points, poses and metadata."""
    torch.manual_seed(0)
    im1 = torch.randn(2, 1, 512, 512, device=device)
    im2 = torch.randn(2, 1, 512, 512, device=device)
    coord1 = torch.rand(2, 50, 2, device=device) * torch.tensor([511., 400.], device=device)
    T1 = torch.eye(4, device=device).unsqueeze(0).repeat(2, 1, 1)
    T2 = torch.eye(4, device=device).unsqueeze(0).repeat(2, 1, 1)
    T2[:, 0, 3] = 0.3
    T2[:, 1, 3] = 0.1
    T2[:, 2, 3] = 0.0
    return {'im1': im1, 'im2': im2, 'coord1': coord1, 'T1': T1, 'T2': T2,
            'meta1': _make_meta(2, device=device),
            'meta2': _make_meta(2, r_max=7.0, device=device)}
