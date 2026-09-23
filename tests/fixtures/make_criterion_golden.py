"""Regenerate the criterion golden fixture from the pre-cleanup implementation.

The fixture pins the numbers the cleaned ``CtoFCriterion`` must reproduce, so the
equivalence test keeps working in the public repository, where the pre-cleanup
commit no longer exists.

Run it from the repository root:

    uv run --no-sync python tests/fixtures/make_criterion_golden.py

Regenerating is a one-time act that needs the pre-cleanup commit in the local
history. The script refuses to run without it, rather than silently writing a
fixture from the current code.
"""
import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINE_COMMIT = '95e6262'
BASELINE_PATH = 'CSONIC/criterion.py'
FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'criterion_golden.npz')

# The knobs the fixture was built with. The test asserts them, so a change here
# without a regeneration is caught rather than silently compared against stale
# numbers.
NUM_SAMPLES = 100
BATCH = 2
N_PTS = 36
W_EPIPOLAR_COARSE = 9.0
W_EPIPOLAR_FINE = 7.0
W_CYCLE_COARSE = 1.0
W_CYCLE_FINE = 1.0  # equal to the coarse weight, so the fixed w_cf typo cannot matter


def make_meta(batch=BATCH, width=512, height=512, r_min=0.1, r_max=10, elev=12, azi=60):
    """Collate the metadata dict the dataloader really produces.

    The dict holds python ints and one float, so ``default_collate`` returns an
    int64/float64 mix. Keeping that mix is the point: it drives the dtype of
    every downstream result.
    """
    meta = {'width': width, 'height': height, 'r_min': r_min,
            'r_max': r_max, 'elev': elev, 'azi': azi}
    collated = default_collate([meta] * batch)
    return {k: v.reshape(-1, 1) for k, v in collated.items()}


def make_inputs():
    """Build the fixture inputs. Procedural and deterministic: no RNG anywhere."""
    batch, n_pts = BATCH, N_PTS
    side = 6
    assert side * side == n_pts

    # A grid of query points. The v range spans near and far, so the arcs of the
    # far points leave the second image (the spatial mask bites) while most of
    # the rest stay inside it.
    u_grid = torch.linspace(40., 470., side)
    v_grid = torch.linspace(160., 460., side)
    uu, vv = torch.meshgrid(u_grid, v_grid, indexing='ij')
    base = torch.stack((uu.reshape(-1), vv.reshape(-1)), dim=-1)
    coord1 = torch.stack([base, base + torch.tensor([7., -11.])], dim=0)

    # Poses: identity, then a 5 degree yaw with a translation in metres.
    T1 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    yaw = torch.deg2rad(torch.tensor(5.0))
    T2 = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    T2[:, 0, 0] = torch.cos(yaw)
    T2[:, 0, 1] = -torch.sin(yaw)
    T2[:, 1, 0] = torch.sin(yaw)
    T2[:, 1, 1] = torch.cos(yaw)
    T2[:, 0, 3] = 0.3
    T2[:, 1, 3] = 0.1
    T2[:, 2, 3] = 1.0

    im2 = make_image(batch)

    # Deterministic offsets for the four predictions, so the cycle terms and the
    # two epipolar terms all differ from each other.
    index = torch.linspace(0., 1., n_pts)
    wobble = torch.stack((torch.cos(6.0 * index), torch.sin(6.0 * index)), dim=-1)
    out = {}
    for key, scale in (('coord2_ec', 4.0), ('coord2_ef', 2.0),
                       ('coord1_lc', 3.0), ('coord1_lf', 1.5)):
        out[key] = coord1 + wobble.unsqueeze(0) * scale
    out.update({'coarse_h': 32, 'coarse_w': 32, 'fine_h': 128, 'fine_w': 128})

    return {'coord1': coord1, 'T1': T1, 'T2': T2, 'im2': im2, 'out': out,
            'meta1': make_meta(r_max=10), 'meta2': make_meta(r_max=7)}


def make_image(batch=BATCH, height=512, width=512):
    """A deterministic bright ramp with two dark bands, so both masks bite."""
    ramp = torch.linspace(0.5, 1.0, width).reshape(1, 1, 1, width)
    im = ramp.repeat(batch, 1, height, 1).clone()
    # Dark stripes inside the lit part, so the intensity mask rejects arcs that
    # the spatial mask accepts.
    for low, high in ((120, 150), (250, 280), (300, 320)):
        im[:, :, low:high, :] = 0.01
    # The shadow band the threshold is measured from.
    im[:, :, 450:, :] = torch.linspace(0.0, 0.05, width).reshape(1, 1, 1, width)
    return im


def make_criterion_args(num_samples=NUM_SAMPLES):
    """The minimal argument object the criterion reads."""
    return argparse.Namespace(
        num_samples=num_samples,
        w_epipolar_coarse=W_EPIPOLAR_COARSE,
        w_epipolar_fine=W_EPIPOLAR_FINE,
        w_cycle_coarse=W_CYCLE_COARSE,
        w_cycle_fine=W_CYCLE_FINE,
        # read by the pre-cleanup implementation only
        w_std=0.0, std=1, range_max=10.0, th_epipolar=1.0, th_cycle=1.0,
    )


def load_baseline(directory):
    """Import the criterion as it was at the pre-cleanup commit."""
    result = subprocess.run(['git', 'show', f'{BASELINE_COMMIT}:{BASELINE_PATH}'],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"Cannot regenerate the fixture: commit {BASELINE_COMMIT} is not in this "
            f"repository, so the pre-cleanup {BASELINE_PATH} cannot be read.\n"
            f"git said: {result.stderr.strip()}\n"
            "Regenerate it from a clone that still has the pre-release history, or "
            "keep the checked-in tests/fixtures/criterion_golden.npz."
        )
    path = os.path.join(directory, 'baseline_criterion.py')
    with open(path, 'w') as handle:
        handle.write(result.stdout)
    spec = importlib.util.spec_from_file_location('baseline_criterion', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    # add_legacy_args lives in the test suite, next to the tests that use it.
    sys.path.insert(0, os.path.join(REPO_ROOT, 'tests'))
    from conftest import add_legacy_args

    with tempfile.TemporaryDirectory() as directory:
        baseline = load_baseline(directory)
    args = add_legacy_args(make_criterion_args())
    criterion = baseline.CtoFCriterion(args)
    data = make_inputs()

    epipolar = criterion.sonar_epipolar_cost(
        data['coord1'], data['out']['coord2_ec'], data['T1'], data['T2'],
        data['im2'], data['im2'], data['meta1'], data['meta2'])
    cycle = criterion.sonar_cycle_consistency_loss(
        data['coord1'], data['out']['coord1_lc'], None, data['im2'], data['im2'],
        data['meta1'], data['meta2'], "coarse")
    loss, eloss_c, eloss_f, closs_c, closs_f, _ = criterion(
        data['coord1'], data['out'], data['T1'], data['T2'], None, data['im2'],
        data['im2'], data['meta1'], data['meta2'])

    if epipolar.numel() == 0:
        raise SystemExit('The epipolar mask rejected every point; the fixture would be empty.')
    if not all(torch.isfinite(t).all() for t in (loss, eloss_c, eloss_f, closs_c, closs_f)):
        raise SystemExit('The baseline produced a non-finite loss; refusing to save the fixture.')

    arrays = {
        'loss': loss, 'eloss_c': eloss_c, 'eloss_f': eloss_f,
        'closs_c': closs_c, 'closs_f': closs_f,
        'epipolar_cost_ec': epipolar, 'cycle_loss_lc': cycle,
    }
    payload = {k: v.detach().cpu().numpy() for k, v in arrays.items()}
    payload['baseline_commit'] = np.array(BASELINE_COMMIT)
    payload['num_samples'] = np.array(NUM_SAMPLES)
    np.savez_compressed(FIXTURE_PATH, **payload)

    print(f'wrote {FIXTURE_PATH} ({os.path.getsize(FIXTURE_PATH)} bytes)')
    for key, value in arrays.items():
        print(f'  {key}: shape={tuple(value.shape)} dtype={value.dtype}')


if __name__ == '__main__':
    sys.exit(main())
