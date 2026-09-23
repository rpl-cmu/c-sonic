"""Regenerate the golden outputs of the released C-SONIC network.

The inputs are procedural, so the test needs no data file and no RNG. Run this
script only when the released checkpoint or the network output changes on
purpose:

    uv run --no-sync python tests/fixtures/make_network_golden.py
"""
import math
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_PATH = os.path.join(REPO_ROOT, 'pretrained', 'CSONIC_pretrained.pth')
GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_golden.npz')
GOLDEN_KEYS = ('coord2_ec', 'coord2_ef', 'coord1_lc', 'coord1_lf',
               'std_c', 'std_f', 'std_lc', 'std_lf')

# the sonar normalization the training dataloader applies
IMAGE_MEAN = 0.052
IMAGE_STD = 0.060


def build_inputs(device='cpu', batch_size=2, height=192, width=192):
    '''
    build a deterministic image pair and query points, with no RNG
    The images are drawn in [0, 1] with sonar-like statistics, dark with sparse
    structure, and then normalized the way the training dataloader normalizes
    them. That puts the activations in the range the network saw during
    training. A raw pattern at full [0, 1] contrast drives large parts of the
    feature maps into denormal floats, which makes a CPU forward ~60x slower.
    A 4 px checkerboard keeps the high frequencies that the fine level needs.
    The probe is 192x192 rather than a full 512x512 frame because half the
    weights of the released checkpoint are denormal floats, which makes a CPU
    forward about 60x slower than the same network with ordinary weights. The
    cost scales with the image area: 512x512 takes ~91 s, 256x256 ~24 s and
    192x192 ~14 s. At 192 the coarse map is 12x12, the fine map 48x48 and the
    fine level window grid 6x6, so nothing degenerates.
    :param device: where to put the tensors
    :return: im1 [B,1,h,w], im2 [B,1,h,w] and coord1 [B,50,2]
    '''
    device = torch.device(device)
    ix = torch.arange(width, dtype=torch.float32)
    iy = torch.arange(height, dtype=torch.float32)
    yy, xx = torch.meshgrid(iy, ix, indexing='ij')
    x = xx / (width - 1.0) * 2.0 - 1.0
    y = yy / (height - 1.0) * 2.0 - 1.0

    # im2 holds the same pattern, rotated by 0.15 rad and shifted
    angle = 0.15
    xr = math.cos(angle) * x - math.sin(angle) * y + 0.12
    yr = math.sin(angle) * x + math.cos(angle) * y - 0.07

    checker1 = (torch.div(xx, 4, rounding_mode='floor')
                + torch.div(yy, 4, rounding_mode='floor')) % 2.0
    checker2 = (torch.div(xx + 9, 4, rounding_mode='floor')
                + torch.div(yy + 5, 4, rounding_mode='floor')) % 2.0

    first, second = [], []
    for b in range(batch_size):
        # a different spatial frequency per sample of the batch
        fx = 3.0 + 1.0 * b
        fy = 5.0 + 2.0 * b
        smooth1 = 0.5 * (torch.sin(fx * x) * torch.cos(fy * y) + 1.0)
        smooth2 = 0.5 * (torch.sin(fx * xr) * torch.cos(fy * yr) + 1.0)
        first.append(0.09 * smooth1 + 0.04 * checker1)
        second.append(0.09 * smooth2 + 0.04 * checker2)
    im1 = (torch.stack(first).unsqueeze(1) - IMAGE_MEAN) / IMAGE_STD
    im2 = (torch.stack(second).unsqueeze(1) - IMAGE_MEAN) / IMAGE_STD

    # 50 query points on a 10 x 5 grid well inside the image.
    # The fractions span [32, 480] of a 512 px frame.
    gx = torch.linspace(0.0625 * width, 0.9375 * width, 10, dtype=torch.float32)
    gy = torch.linspace(0.0625 * height, 0.9375 * height, 5, dtype=torch.float32)
    gyy, gxx = torch.meshgrid(gy, gx, indexing='ij')
    grid = torch.stack((gxx.reshape(-1), gyy.reshape(-1)), dim=-1)  # 50x2, (x, y)
    # offset each sample of the batch so the two are not identical
    coord1 = torch.stack([grid + 0.0137 * width * b for b in range(batch_size)])
    return im1.to(device), im2.to(device), coord1.to(device)


def build_released_net(device='cpu'):
    '''
    build the released cross attention network and load its weights
    The configuration is the one the released checkpoint was trained with, so
    the golden outputs do not move when the test fixtures change.
    :param device: where to put the network
    :return: the network in eval mode
    '''
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    import config
    from CSONIC.network import CSONICNet

    try:
        args = config.get_args([])
    except TypeError:
        args = config.get_args()
    args.cross_attention = 1
    args.pretrained = 0
    args.backbone = 'resnet34'
    args.coarse_feat_dim = 64
    args.fine_feat_dim = 64

    net = CSONICNet(args, torch.device(device))
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    net.load_state_dict(ckpt['state_dict'], strict=True)
    net.to(torch.device(device))
    net.eval()
    return net


def main():
    if not os.path.exists(CKPT_PATH):
        raise SystemExit('missing the released checkpoint at %s' % CKPT_PATH)
    net = build_released_net('cpu')
    im1, im2, coord1 = build_inputs('cpu')
    print('im1 range [%.3f, %.3f] mean %.3f std %.3f'
          % (im1.min(), im1.max(), im1.mean(), im1.std()))
    with torch.no_grad():
        out = net(im1, im2, coord1)
    arrays = {k: out[k].cpu().numpy() for k in GOLDEN_KEYS}
    np.savez_compressed(GOLDEN_PATH, **arrays)
    print('wrote %s (%.1f KiB)' % (GOLDEN_PATH, os.path.getsize(GOLDEN_PATH) / 1024.0))
    for k in GOLDEN_KEYS:
        print('  %-10s %-12s min %10.4f max %10.4f' % (
            k, str(arrays[k].shape), arrays[k].min(), arrays[k].max()))


if __name__ == '__main__':
    main()
