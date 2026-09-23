"""End-to-end checks on the released checkpoint.

These load the full network, so the whole module is marked slow. Point
CSONIC_CKPT at your own checkpoint to run them against it.
"""
import os

import numpy as np
import pytest
import torch

import CSONIC_test
import detection

pytestmark = pytest.mark.slow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUPERPOINT_WEIGHTS = os.path.join(REPO_ROOT, 'pretrained', 'superpoint_v1.pth')

# The real-world sample pairs. Set CSONIC_SAMPLE_DIR to the directory holding
# them; the tests that need real sonar images skip when it is not set.
SAMPLE_DIR = os.environ.get('CSONIC_SAMPLE_DIR')

# The size and the window the released model was trained with.
IMAGE_SIZE = 512
WINDOW_SIZE = 0.125
# coord2_ef is the centre of a search window, so it can sit this far outside.
MARGIN = WINDOW_SIZE * (IMAGE_SIZE - 1) / 2


def _find_checkpoint():
    """Return the checkpoint to test, or None when there is nothing to test."""
    from_env = os.environ.get('CSONIC_CKPT')
    if from_env:
        return from_env if os.path.isfile(from_env) else None
    for name in ('pretrained/CSONIC_pretrained.pth', 'best_csonic.pth'):
        path = os.path.join(REPO_ROOT, name)
        if os.path.isfile(path):
            return path
    return None


CKPT = _find_checkpoint()
if CKPT is None:
    pytest.skip('no checkpoint found; set CSONIC_CKPT to one',
                allow_module_level=True)


@pytest.fixture(scope='module')
def model():
    """The released model, loaded once for the whole module."""
    return CSONIC_test.load_model(CKPT)


def _procedural_pair():
    """A deterministic image pair and its query points. No RNG involved."""
    rows, cols = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    base = (np.sin(rows / 17.0) * np.cos(cols / 23.0) + 1.0) * 0.5
    im1 = (base * 255).astype(np.uint8)
    im2 = np.roll(im1, 7, axis=1)
    coord = np.stack([np.linspace(50, 460, 40), np.linspace(60, 450, 40)], axis=1)
    return im1, im2, coord


def _sample(rel):
    if not SAMPLE_DIR:
        pytest.skip('set CSONIC_SAMPLE_DIR to the directory holding the sonar '
                    'samples to run this test')
    path = os.path.join(SAMPLE_DIR, rel)
    if not os.path.isfile(path):
        pytest.skip('sample not found: {}; check CSONIC_SAMPLE_DIR'.format(path))
    return path


def _real_pair():
    """Paths of one cross-modal pair: H0/3 against L0/4."""
    if not os.path.isfile(SUPERPOINT_WEIGHTS):
        pytest.skip('SuperPoint weights not found at {}'.format(SUPERPOINT_WEIGHTS))
    return (_sample('H0/3.png'), _sample('L0/4.png'),
            _sample('H0/3_meta.npy'), _sample('L0/4_meta.npy'),
            _sample('H0/3.npy'), _sample('L0/4.npy'))


def test_checkpoint_loads_strict(model):
    """Every weight in the file belongs to the network, and none is missing.

    The model fixture is shared by the whole module, and a strict load puts the
    raw, subnormal weights back into it, so the weights are restored afterwards.
    """
    assert model.start_step == 160000

    raw = torch.load(CKPT, map_location='cpu', weights_only=True)
    saved = {name: value.detach().cpu().clone()
             for name, value in model.model.state_dict().items()}
    try:
        result = model.model.load_state_dict(raw['state_dict'], strict=True)
    finally:
        model.model.load_state_dict(saved, strict=True)

    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []
    assert len(raw['state_dict']) > 0


def test_checkpoint_forward_finite_on_procedural_pair(model):
    """A forward pass on a fixed pair gives finite predictions in frame."""
    im1, im2, coord = _procedural_pair()
    t1 = CSONIC_test.prepare_image(im1, model.device)
    t2 = CSONIC_test.prepare_image(im2, model.device)
    kpts = torch.from_numpy(coord).float().unsqueeze(0).to(model.device)

    model.model.eval()
    with torch.no_grad():
        out = model.model(t1, t2, kpts)

    for key in ('coord2_ec', 'coord2_ef', 'coord1_lc', 'coord1_lf'):
        value = out[key].detach().cpu().numpy()
        assert value.shape == (1, len(coord), 2), key
        assert np.all(np.isfinite(value)), key
    for key in ('std_c', 'std_f', 'std_lc', 'std_lf'):
        value = out[key].detach().cpu().numpy()
        assert value.shape == (1, len(coord)), key
        assert np.all(np.isfinite(value)) and np.all(value >= 0), key

    coord2_ef = out['coord2_ef'].detach().cpu().numpy()[0]
    assert coord2_ef.min() >= -MARGIN
    assert coord2_ef.max() <= IMAGE_SIZE - 1 + MARGIN
    # The prediction must depend on the images, not just echo the query points.
    assert np.abs(coord2_ef - coord).max() > 1e-3


def test_checkpoint_loads_on_cpu(monkeypatch):
    """The checkpoint loads without a GPU, and its weights are cleaned there."""
    tiny = torch.finfo(torch.float32).tiny
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    model = CSONIC_test.load_model(CKPT)

    assert model.device == 'cpu'
    assert model.start_step == 160000
    assert not model.model.training
    for name, value in model.model.state_dict().items():
        if value.is_floating_point():
            assert int(((value.abs() > 0) & (value.abs() < tiny)).sum()) == 0, name


def test_load_model_builds_everything_on_the_requested_device():
    """device='cpu' builds on the CPU and allocates nothing on the GPU.

    Building on one device and moving afterwards leaves the optimizer state and
    the criterion where they started, and on a CUDA machine it allocates GPU
    memory the caller did not ask for.
    """
    before = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

    model = CSONIC_test.load_model(CKPT, device='cpu')

    assert model.device == 'cpu'
    # The criterion holds no tensors today, so it cannot catch a stray device
    # on its own; it is checked here so that it does once it gains one.
    tensors = list(model.model.parameters()) + list(model.model.buffers()) \
        + list(model.criterion.parameters()) + list(model.criterion.buffers())
    assert len(tensors) > 0
    for tensor in tensors:
        assert tensor.device.type == 'cpu'
    assert model.model.device == 'cpu'

    if before is not None:
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == before
        # Nothing was allocated on the GPU even briefly. Building there and
        # moving afterwards frees the memory again, so only the peak shows it.
        assert torch.cuda.max_memory_allocated() <= before


def test_inference_skips_the_optimizer_state_but_training_still_loads_it():
    """Reading the optimizer state back is for resuming training, not inference.

    Inference never steps the optimizer, so loading its state would allocate a
    second copy of every parameter for nothing. A training run still needs it.
    """
    import config
    from CSONIC.csonic_model import CSONICModel

    raw = torch.load(CKPT, map_location='cpu', weights_only=True)
    assert raw['optimizer']['state'], 'the checkpoint carries no optimizer state'

    def build(phase):
        args = config.get_args([])
        args.ckpt_path = CKPT
        args.pretrained = 0
        args.phase = phase
        return CSONICModel(args, device='cpu')

    for_inference = build('test')
    for_training = build('train')

    assert for_inference.start_step == for_training.start_step == 160000
    assert for_inference.optimizer.state_dict()['state'] == {}
    assert for_training.optimizer.state_dict()['state']
    assert for_training.scheduler.state_dict()['last_epoch'] == \
        raw['scheduler']['last_epoch']


def test_load_model_leaves_the_rng_state_and_the_denormal_flag_alone(monkeypatch):
    """Loading a model changes nothing process wide.

    Building the network draws from the global generators to initialise weights
    the checkpoint then overwrites, which would silently shift a caller's own
    sequence of random numbers. And torch.set_flush_denormal would change the
    arithmetic of every other library in the process; it only reaches threads
    started after it, and measured on this checkpoint it made the forward pass
    slower rather than faster when set after the thread pool existed.
    """
    calls = []
    monkeypatch.setattr(torch, 'set_flush_denormal', lambda flag: calls.append(flag) or True)
    expected = CSONIC_test._default_device()

    torch.manual_seed(1234)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None

    on_default = CSONIC_test.load_model(CKPT)

    assert torch.equal(torch.get_rng_state(), cpu_before)
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_before)

    # The CPU is where the flag used to be set, so check that path too.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    on_cpu = CSONIC_test.load_model(CKPT)

    assert on_default.device == expected
    assert on_cpu.device == 'cpu'
    assert calls == []


def test_expectation_matching_real_pair(model, capsys):
    """The released model matches a real cross-modal pair, H0/3 against L0/4."""
    img1, img2, meta1, meta2, pose1, pose2 = _real_pair()

    mkpts0, mkpts1, std_f = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model=model, th=1.0,
        superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(mkpts0) == len(mkpts1) == len(std_f)
    assert len(mkpts0) >= 5
    assert mkpts0.shape[1] == 2 and mkpts1.shape[1] == 2
    assert np.all(np.isfinite(mkpts1))
    assert mkpts1.min() >= -MARGIN
    assert mkpts1.max() <= IMAGE_SIZE - 1 + MARGIN
    # The query points come from image 1 and stay inside it.
    assert mkpts0.min() >= 0 and mkpts0.max() < IMAGE_SIZE
    # std_f is returned sorted, smallest first.
    assert np.all(np.diff(std_f) >= -1e-6)

    meta1_d = detection.read_meta(meta1)
    meta2_d = detection.read_meta(meta2)
    projected = CSONIC_test.project_keypoints(
        mkpts0, np.load(pose1), np.load(pose2), meta1_d, meta2_d)
    idx, percentage, mean_dist, std_dist = CSONIC_test.inlier_percentage(
        projected, mkpts1, px_th=20)

    assert 0.0 <= percentage <= 100.0
    assert len(idx) == int(round(percentage / 100.0 * len(mkpts0)))
    assert np.isfinite(mean_dist) and np.isfinite(std_dist)
    with capsys.disabled():
        print('\nH0/3 -> L0/4: {} matches, inliers {:.1f}%, mean {:.1f} px'
              .format(len(mkpts0), percentage, mean_dist))


def test_expectation_matching_loads_the_model_from_a_path(model):
    """Given model_path instead of a model, the checkpoint is loaded here."""
    img1, img2, meta1, meta2, _, _ = _real_pair()

    from_path = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model_path=CKPT, th=1.0,
        superpoint_weights=SUPERPOINT_WEIGHTS)
    from_model = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model=model, th=1.0,
        superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(from_path[0]) > 0
    for loaded, given in zip(from_path, from_model):
        assert np.allclose(loaded, given, atol=1e-5)


def test_expectation_matching_threshold_filters(model):
    """A tight threshold keeps fewer matches than a loose one."""
    img1, img2, meta1, meta2, _, _ = _real_pair()

    _, _, loose = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model=model, th=1.0,
        superpoint_weights=SUPERPOINT_WEIGHTS)
    _, _, tight = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model=model, th=float(np.median(loose)),
        superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(tight) < len(loose)
    assert np.all(tight < np.median(loose))


def test_expectation_matching_roi_restricts_the_query_points(model):
    """kpt_roi keeps only the query points inside the given box."""
    img1, img2, meta1, meta2, _, _ = _real_pair()
    roi = (0, 512, 220, 400)

    mkpts0, _, _ = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model=model, th=1.0, kpt_roi=roi,
        superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(mkpts0) > 0
    assert mkpts0[:, 0].min() >= roi[0] and mkpts0[:, 0].max() <= roi[1]
    assert mkpts0[:, 1].min() >= roi[2] and mkpts0[:, 1].max() <= roi[3]


def test_legacy_input_path_runs(model):
    """The paper-era pipeline still runs: raw pixel values, dropout active."""
    img1, img2, meta1, meta2, _, _ = _real_pair()

    mkpts0, mkpts1, std_f = CSONIC_test.expectation_matching(
        img1, img2, meta1, meta2, model=model, th=1.0, normalize_input=False,
        superpoint_weights=SUPERPOINT_WEIGHTS)

    assert len(mkpts0) == len(mkpts1) == len(std_f)
    assert np.all(np.isfinite(mkpts1))
    # The model is left in eval mode for whoever runs next.
    assert not model.model.training


def test_legacy_input_leaves_the_model_as_it_found_it(model):
    """A legacy run must not change the batch norm statistics, or later runs.

    Batch normalisation folds the batch statistics into its running averages in
    train mode, even under no_grad. If inference let that happen, every default
    run after a legacy run would use disturbed averages and return different
    matches.
    """
    img1, img2, meta1, meta2, _, _ = _real_pair()
    call = dict(model=model, th=1.0, superpoint_weights=SUPERPOINT_WEIGHTS)

    _, before, _ = CSONIC_test.expectation_matching(img1, img2, meta1, meta2, **call)
    stats = {name: value.clone()
             for name, value in model.model.state_dict().items()
             if 'running_' in name or 'num_batches_tracked' in name}
    assert len(stats) > 0

    CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                     normalize_input=False, **call)

    for name, value in stats.items():
        assert torch.equal(value, model.model.state_dict()[name]), name
    _, after, _ = CSONIC_test.expectation_matching(img1, img2, meta1, meta2, **call)
    assert np.array_equal(before, after)


def test_batch_mode_is_deterministic(model):
    """bn_mode='batch' has no dropout in it, so two runs agree bit for bit."""
    img1, img2, meta1, meta2, _, _ = _real_pair()
    call = dict(model=model, th=1.0, bn_mode='batch',
                superpoint_weights=SUPERPOINT_WEIGHTS)

    first = CSONIC_test.expectation_matching(img1, img2, meta1, meta2, **call)
    second = CSONIC_test.expectation_matching(img1, img2, meta1, meta2, **call)

    assert len(first[0]) > 0
    for a, b in zip(first, second):
        assert np.array_equal(a, b)


def test_batch_mode_leaves_the_running_statistics_untouched(model):
    """Per-image statistics must not be folded into the stored averages."""
    img1, img2, meta1, meta2, _, _ = _real_pair()
    call = dict(model=model, th=1.0, superpoint_weights=SUPERPOINT_WEIGHTS)

    _, before, _ = CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                                    bn_mode='running', **call)
    stats = {name: value.clone()
             for name, value in model.model.state_dict().items()
             if 'running_' in name or 'num_batches_tracked' in name}
    assert len(stats) > 0

    CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                     bn_mode='batch', **call)

    for name, value in stats.items():
        assert torch.equal(value, model.model.state_dict()[name]), name
    _, after, _ = CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                                   bn_mode='running', **call)
    assert np.array_equal(before, after)


def test_batch_mode_runs_batchnorm_in_train_mode_and_dropout_in_eval(model):
    """The middle path: batch statistics, no dropout.

    Recorded from inside the forward pass, so it pins what the modules actually
    did rather than what was asked for.
    """
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.dropout import _DropoutNd

    img1, img2, meta1, meta2, _, _ = _real_pair()
    seen = {'bn': [], 'drop': []}
    handles = []
    for module in model.model.modules():
        if isinstance(module, _BatchNorm):
            handles.append(module.register_forward_hook(
                lambda m, i, o: seen['bn'].append((m.training, m.track_running_stats))))
        elif isinstance(module, _DropoutNd):
            handles.append(module.register_forward_hook(
                lambda m, i, o: seen['drop'].append(m.training)))
    assert handles, 'the network has neither batch norm nor dropout'

    try:
        CSONIC_test.expectation_matching(
            img1, img2, meta1, meta2, model=model, th=1.0, bn_mode='batch',
            superpoint_weights=SUPERPOINT_WEIGHTS)
    finally:
        for handle in handles:
            handle.remove()

    assert seen['bn'], 'no batch norm layer ran'
    assert all(training for training, _ in seen['bn'])
    assert not any(track for _, track in seen['bn'])
    assert seen['drop'], 'no dropout layer ran'
    assert not any(seen['drop'])


def test_running_mode_keeps_every_layer_in_eval(model):
    """bn_mode='running' is plain eval: no layer is in train mode."""
    from torch.nn.modules.batchnorm import _BatchNorm

    img1, img2, meta1, meta2, _, _ = _real_pair()
    seen = []
    handles = [m.register_forward_hook(lambda mod, i, o: seen.append(mod.training))
               for m in model.model.modules() if isinstance(m, _BatchNorm)]
    try:
        CSONIC_test.expectation_matching(
            img1, img2, meta1, meta2, model=model, th=1.0, bn_mode='running',
            superpoint_weights=SUPERPOINT_WEIGHTS)
    finally:
        for handle in handles:
            handle.remove()

    assert seen and not any(seen)


def test_the_two_bn_modes_disagree(model):
    """The modes are not the same computation, or the flag would be pointless."""
    img1, img2, meta1, meta2, _, _ = _real_pair()
    call = dict(model=model, th=1.0, superpoint_weights=SUPERPOINT_WEIGHTS)

    _, running, _ = CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                                     bn_mode='running', **call)
    _, batch, _ = CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                                   bn_mode='batch', **call)

    assert running.shape == batch.shape
    assert np.abs(running - batch).max() > 1.0


def test_released_checkpoint_has_subnormal_weights():
    """The zeroing below is not a no-op: the stored file really has them."""
    tiny = torch.finfo(torch.float32).tiny
    raw = torch.load(CKPT, map_location='cpu', weights_only=True)['state_dict']

    subnormal = sum(int(((value.abs() > 0) & (value.abs() < tiny)).sum())
                    for value in raw.values() if value.is_floating_point())

    assert subnormal > 0


def test_load_model_leaves_no_subnormal_weights(model):
    """Nothing in the loaded model is small enough to fall off the fast path."""
    tiny = torch.finfo(torch.float32).tiny

    for name, value in model.model.state_dict().items():
        if not value.is_floating_point():
            continue
        assert int(((value.abs() > 0) & (value.abs() < tiny)).sum()) == 0, name


def test_csonic_model_zeroes_the_subnormals_it_loads():
    """The notebook builds CSONICModel itself, and it gets clean weights too."""
    import config
    from CSONIC.csonic_model import CSONICModel

    tiny = torch.finfo(torch.float32).tiny
    raw = torch.load(CKPT, map_location='cpu', weights_only=True)['state_dict']
    stored = sum(int(((value.abs() > 0) & (value.abs() < tiny)).sum())
                 for value in raw.values() if value.is_floating_point())

    # The first cell of Jupyter/Visualizer.ipynb sets these.
    args = config.get_args([])
    args.ckpt_path = CKPT
    args.pretrained = 0
    args.phase = 'test'
    model = CSONICModel(args, device='cpu')

    assert model.zeroed_subnormal == stored > 0
    for name, value in model.model.state_dict().items():
        if value.is_floating_point():
            assert int(((value.abs() > 0) & (value.abs() < tiny)).sum()) == 0, name


def test_zeroing_subnormals_keeps_the_subnormals_when_asked_not_to():
    """The switch really controls it, so the test above is not vacuous."""
    tiny = torch.finfo(torch.float32).tiny
    untouched = CSONIC_test.load_model(CKPT, zero_subnormal_weights=False)

    subnormal = sum(int(((value.abs() > 0) & (value.abs() < tiny)).sum())
                    for value in untouched.model.state_dict().values()
                    if value.is_floating_point())

    assert subnormal > 0


def test_zeroing_subnormals_does_not_change_the_matches(model):
    """A weight below 1.2e-38 cannot move a sum of normal numbers."""
    img1, img2, meta1, meta2, _, _ = _real_pair()
    untouched = CSONIC_test.load_model(CKPT, zero_subnormal_weights=False)
    call = dict(th=1.0, bn_mode='running', superpoint_weights=SUPERPOINT_WEIGHTS)

    zeroed_out = CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                                  model=model, **call)
    kept_out = CSONIC_test.expectation_matching(img1, img2, meta1, meta2,
                                                model=untouched, **call)

    assert len(zeroed_out[0]) > 0
    for zeroed, kept in zip(zeroed_out, kept_out):
        assert np.abs(zeroed - kept).max() == 0.0


def test_main_with_no_matches_still_writes_a_figure(tmp_path):
    """A threshold that keeps nothing must not crash the command line."""
    img1, img2, meta1, meta2, pose1, pose2 = _real_pair()
    out = tmp_path / 'figures' / 'match.png'

    CSONIC_test.main([
        '--img1', img1, '--img2', img2, '--meta1', meta1, '--meta2', meta2,
        '--pose1', pose1, '--pose2', pose2, '--model', CKPT,
        '--th', '0', '--ransac', '--seed', '0', '--out', str(out),
        '--superpoint-weights', SUPERPOINT_WEIGHTS])

    assert out.is_file() and out.stat().st_size > 0


def test_prepare_image_normalizes_like_the_training_transform():
    """prepare_image reproduces ToTensor followed by Normalize."""
    from dataloader.sonardata import SONAR_MEAN, SONAR_STD

    rng = np.random.default_rng(8)
    img = rng.integers(0, 256, size=(64, 48), dtype=np.uint8)

    normalized = CSONIC_test.prepare_image(img, 'cpu')
    raw = CSONIC_test.prepare_image(img, 'cpu', normalize=False)

    assert normalized.shape == (1, 1, 64, 48)
    assert normalized.dtype == torch.float32
    want = (img.astype(np.float32) / 255.0 - SONAR_MEAN) / SONAR_STD
    assert np.allclose(normalized.numpy()[0, 0], want, atol=1e-6)
    # The legacy path feeds the stored pixel values, unscaled.
    assert np.allclose(raw.numpy()[0, 0], img.astype(np.float32))


def test_cpu_load_never_asks_cuda_which_device_is_current(monkeypatch):
    """A CPU load must not initialise CUDA, not even to pick the generator to fork."""
    def boom(*args, **kwargs):
        raise AssertionError('torch.cuda.current_device() was called during a CPU load')
    monkeypatch.setattr(torch.cuda, 'current_device', boom)
    model = CSONIC_test.load_model(CKPT, device='cpu')
    assert model.device == 'cpu'
    assert all(p.device.type == 'cpu' for p in model.model.parameters())
