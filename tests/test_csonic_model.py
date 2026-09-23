"""Tests for the CSONICModel wrapper: checkpoint loading and batch unpacking."""
import copy

import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from CSONIC import network
from CSONIC.csonic_model import CSONICModel
from CSONIC.network import CSONICNet


def make_sample(seed, r_max=10):
    """Build one item with the keys and python types SonarData returns."""
    generator = torch.Generator().manual_seed(seed)
    return {
        'im1': torch.rand(1, 64, 64, generator=generator),
        'im2': torch.rand(1, 64, 64, generator=generator),
        'pose': torch.eye(4),
        'T1': torch.eye(4),
        'T2': torch.eye(4),
        'coord1': torch.rand(5, 2, generator=generator),
        'im1_width': 512, 'im1_height': 512,
        'im2_width': 512, 'im2_height': 512,
        'im1_r_min': 0.1, 'im1_r_max': 10,
        'im2_r_min': 0.1, 'im2_r_max': r_max,
        'im1_elev': 12, 'im1_azi': 60,
        'im2_elev': 12, 'im2_azi': 60,
    }


def save_throwaway_ckpt(args, path, step):
    """Write a checkpoint holding a freshly initialised network, no downloads."""
    build_args = copy.deepcopy(args)
    build_args.pretrained = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'step': step, 'state_dict': CSONICNet(build_args, 'cpu').state_dict()},
               path)


def record_pretrained(monkeypatch):
    """Patch build_resnet so the test sees the pretrained flag it receives."""
    recorded = []
    original_build_resnet = network.build_resnet

    def recording_build_resnet(encoder, pretrained):
        recorded.append(pretrained)
        return original_build_resnet(encoder, pretrained)

    monkeypatch.setattr(network, 'build_resnet', recording_build_resnet)
    return recorded


@pytest.mark.slow
def test_ckpt_path_skips_imagenet_download(tmp_path, args, monkeypatch):
    """An explicit checkpoint supplies every weight, so ImageNet is skipped."""
    args.outdir = str(tmp_path)
    ckpt = tmp_path / 'ckpt.pth'
    save_throwaway_ckpt(args, ckpt, step=7)

    recorded = record_pretrained(monkeypatch)
    args.ckpt_path = str(ckpt)
    args.pretrained = 1
    model = CSONICModel(args)

    # The backbone really went through the hook, so the assertion below is real.
    assert recorded
    assert not any(recorded), 'the backbone asked for ImageNet weights: {}'.format(recorded)
    assert model.start_step == 7
    # The caller's namespace is not mutated; only the copy the network reads is.
    assert args.pretrained == 1


@pytest.mark.slow
def test_auto_resume_skips_imagenet_download(tmp_path, args, monkeypatch):
    """Resuming from outdir also skips ImageNet, not only an explicit path."""
    args.outdir = str(tmp_path)
    save_throwaway_ckpt(args, tmp_path / args.exp_name / '000007.pth', step=7)

    recorded = record_pretrained(monkeypatch)
    args.ckpt_path = ''
    args.pretrained = 1
    model = CSONICModel(args)

    assert recorded
    assert not any(recorded), 'the backbone asked for ImageNet weights: {}'.format(recorded)
    assert model.start_step == 7
    assert args.pretrained == 1


@pytest.mark.slow
def test_no_checkpoint_keeps_imagenet_weights(tmp_path, args, monkeypatch):
    """With nothing to resume from, the ImageNet request is left alone.

    The stub records the flag and then builds the same backbone unweighted, so
    the test never downloads anything.
    """
    args.outdir = str(tmp_path)
    recorded = []
    original_build_resnet = network.build_resnet

    def stub_build_resnet(encoder, pretrained):
        recorded.append(pretrained)
        return original_build_resnet(encoder, False)

    monkeypatch.setattr(network, 'build_resnet', stub_build_resnet)
    args.ckpt_path = ''
    args.pretrained = 1
    model = CSONICModel(args)

    assert recorded == [1]
    assert model.start_step == 0


@pytest.mark.slow
def test_missing_ckpt_path_raises(tmp_path, args):
    """A --ckpt_path that does not exist fails before the network is built."""
    args.outdir = str(tmp_path)
    args.ckpt_path = str(tmp_path / 'nope.pth')
    with pytest.raises(Exception) as excinfo:
        CSONICModel(args)
    assert 'no checkpoint found' in str(excinfo.value)


@pytest.mark.slow
def test_set_input_builds_meta_and_drops_pose(tmp_path, args):
    """set_input splits the batch into two metadata dicts and ignores the pose."""
    args.pretrained = 0
    args.ckpt_path = ''
    args.outdir = str(tmp_path)
    model = CSONICModel(args)

    batch = default_collate([make_sample(0), make_sample(1, r_max=7)])
    # The batch really carries a pose, so dropping it is a choice, not an absence.
    assert 'pose' in batch
    model.set_input(batch)

    assert model.meta1['r_max'].shape == (2, 1)
    assert model.meta2['r_max'].shape == (2, 1)
    # default_collate gives an int64/float64 mix; the criterion depends on it.
    assert model.meta1['width'].dtype == torch.int64
    assert model.meta1['r_min'].dtype == torch.float64
    # meta2 reads the im2_ keys, not the im1_ ones.
    assert model.meta2['r_max'].flatten().tolist() == [10, 7]
    assert model.meta1['r_max'].flatten().tolist() == [10, 10]

    assert not hasattr(model, 'pose')
    assert model.im1.shape == (2, 1, 64, 64)
    assert model.coord1.shape == (2, 5, 2)
    assert model.T1.shape == (2, 4, 4)


# --------------------------------------------------------------------------
# subnormal weights
# --------------------------------------------------------------------------

# The smallest normal float32. A nonzero value below it is subnormal.
TINY = torch.finfo(torch.float32).tiny
# One subnormal of each sign.
POSITIVE_SUBNORMAL = 1e-40
NEGATIVE_SUBNORMAL = -3e-41
# Where the checkpoint below puts them: a convolution weight, a float buffer of
# batch normalisation, and the integer buffer next to it.
CONV_WEIGHT = 'net_prep.layer1.0.conv1.weight'
BN_BUFFER = 'net_prep.firstbn.running_var'
BN_COUNTER = 'net_prep.firstbn.num_batches_tracked'
INJECTED_SUBNORMALS = 4


def _subnormal(tensor):
    return (tensor.abs() > 0) & (tensor.abs() < TINY)


def _bits(tensor):
    """The raw bit pattern of a float32 tensor, so -0.0 differs from 0.0."""
    assert tensor.dtype == torch.float32
    return tensor.detach().contiguous().view(torch.int32)


def _count_subnormals(state):
    return sum(int(_subnormal(value).sum()) for value in state.values()
               if value.is_floating_point())


def _set(flat, values):
    for index, value in enumerate(values):
        flat[index] = value


def save_ckpt_with_subnormals(args, path, step=7):
    """Save a checkpoint laid out like save_model, with known subnormals.

    The weights hold two subnormals in a convolution weight and two in a float
    buffer of batch normalisation, next to exact zeros of both signs, the
    smallest normal float and ordinary values. The optimizer has taken one step,
    so its state is not empty, and one entry of that state is subnormal too.

    :return: the dict that was saved
    """
    build_args = copy.deepcopy(args)
    build_args.pretrained = 0
    build_args.ckpt_path = ''
    build_args.phase = 'train'
    build_args.outdir = str(path.parent / 'source')
    source = CSONICModel(build_args, device='cpu')
    for param in source.model.parameters():
        param.grad = torch.ones_like(param)
    source.optimizer.step()
    source.scheduler.step()

    state = {name: value.clone() for name, value in source.model.state_dict().items()}
    _set(state[CONV_WEIGHT].view(-1), [POSITIVE_SUBNORMAL, NEGATIVE_SUBNORMAL, 0.0,
                                       -0.0, TINY, -TINY, 0.5, -0.25])
    _set(state[BN_BUFFER].view(-1), [POSITIVE_SUBNORMAL, 0.0, NEGATIVE_SUBNORMAL, 2.0])
    state[BN_COUNTER].fill_(12345)

    optimizer = copy.deepcopy(source.optimizer.state_dict())
    optimizer['state'][0]['exp_avg'].view(-1)[0] = POSITIVE_SUBNORMAL

    saved = {'step': step, 'state_dict': state, 'optimizer': optimizer,
             'scheduler': copy.deepcopy(source.scheduler.state_dict())}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    # The injection is what the tests count; the fresh network adds none.
    assert _count_subnormals(state) == INJECTED_SUBNORMALS
    return saved


def assert_zeroed_like(loaded, saved):
    """No subnormal is left; every other value keeps its exact bits."""
    for name, before in saved.items():
        after = loaded[name]
        if not before.is_floating_point():
            assert torch.equal(after, before), name
            continue
        assert int(_subnormal(after).sum()) == 0, name
        mask = _subnormal(before)
        assert torch.equal(_bits(after)[~mask], _bits(before)[~mask]), name
        assert torch.equal(_bits(after)[mask], torch.zeros_like(_bits(after)[mask])), name


def test_zero_subnormal_weights_zeroes_only_the_subnormal_floats():
    """Both signs, in parameters and in buffers; the rest keeps its bits."""
    from CSONIC import csonic_model

    net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.BatchNorm2d(2))
    with torch.no_grad():
        _set(net[0].weight.view(-1), [POSITIVE_SUBNORMAL, NEGATIVE_SUBNORMAL, 0.0,
                                      -0.0, TINY, 0.5])
        _set(net[1].running_mean, [NEGATIVE_SUBNORMAL, 1.0])
    net[1].num_batches_tracked.fill_(3)
    before = {name: value.clone() for name, value in net.state_dict().items()}
    weight = net[0].weight

    zeroed = csonic_model.zero_subnormal_weights(net)

    assert zeroed == 3
    assert_zeroed_like(net.state_dict(), before)
    # In place: the optimizer keeps pointing at the same parameters.
    assert net[0].weight is weight and weight.requires_grad
    assert int(net[1].num_batches_tracked) == 3
    assert csonic_model.zero_subnormal_weights(net) == 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32,
                                   torch.float64])
def test_zero_subnormal_weights_uses_the_limit_of_each_dtype(dtype):
    """A value below the smallest normal of its own dtype goes; the rest stays.

    The float32 limit would miss the float16 subnormals, and it would zero
    normal float64 values between the float64 limit and the float32 limit.
    """
    from CSONIC import csonic_model

    info = torch.finfo(dtype)
    subnormal = info.tiny / 4
    values = [subnormal, -subnormal, info.tiny, -info.tiny, 0.0, 1.0]
    if dtype == torch.float64:
        # Normal in float64, but below the smallest normal float32.
        values.append(torch.finfo(torch.float32).tiny / 1000)
    net = torch.nn.Module()
    net.weight = torch.nn.Parameter(torch.tensor(values, dtype=dtype))
    net.register_buffer('state', torch.tensor(values, dtype=dtype))
    assert net.weight[0] != 0, 'the subnormal test value must be representable'

    zeroed = csonic_model.zero_subnormal_weights(net)

    expected = torch.tensor([0.0, 0.0] + values[2:], dtype=dtype)
    assert zeroed == 4
    assert torch.equal(net.weight.detach(), expected)
    assert torch.equal(net.state, expected)


def test_loading_a_checkpoint_zeroes_its_subnormal_weights(tmp_path, args):
    """The model loader itself cleans the weights, so every caller gets it."""
    ckpt = tmp_path / 'ckpt.pth'
    saved = save_ckpt_with_subnormals(args, ckpt)
    args.outdir = str(tmp_path)
    args.ckpt_path = str(ckpt)
    args.phase = 'test'

    model = CSONICModel(args, device='cpu')

    assert model.start_step == 7
    assert model.zeroed_subnormal == INJECTED_SUBNORMALS
    assert_zeroed_like(model.model.state_dict(), saved['state_dict'])
    assert int(model.model.state_dict()[BN_COUNTER]) == 12345


def test_loading_keeps_the_subnormal_weights_when_asked_not_to(tmp_path, args):
    """With the switch off, the model holds the stored values, bit for bit."""
    ckpt = tmp_path / 'ckpt.pth'
    saved = save_ckpt_with_subnormals(args, ckpt)
    args.outdir = str(tmp_path)
    args.ckpt_path = str(ckpt)
    args.phase = 'test'

    model = CSONICModel(args, device='cpu', zero_subnormal_weights=False)

    assert model.zeroed_subnormal == 0
    loaded = model.model.state_dict()
    assert loaded.keys() == saved['state_dict'].keys()
    for name, before in saved['state_dict'].items():
        after = loaded[name]
        if before.is_floating_point():
            assert torch.equal(_bits(after), _bits(before)), name
        else:
            assert torch.equal(after, before), name
    assert _count_subnormals(loaded) == INJECTED_SUBNORMALS


@pytest.mark.parametrize('how', ['resume', 'ckpt_path'])
def test_resuming_training_zeroes_the_weights_and_keeps_the_optimizer(tmp_path, args, how):
    """train.py resumes through the same loader, and the optimizer is untouched."""
    args.outdir = str(tmp_path)
    args.phase = 'train'
    if how == 'resume':
        ckpt = tmp_path / args.exp_name / '000007.pth'
        args.ckpt_path = ''
    else:
        ckpt = tmp_path / 'elsewhere' / 'ckpt.pth'
        args.ckpt_path = str(ckpt)
    saved = save_ckpt_with_subnormals(args, ckpt)

    model = CSONICModel(args, device='cpu')

    assert model.start_step == 7
    assert model.zeroed_subnormal == INJECTED_SUBNORMALS
    assert_zeroed_like(model.model.state_dict(), saved['state_dict'])

    # The optimizer state came back as stored, its subnormal included.
    state = model.optimizer.state_dict()['state']
    assert state.keys() == saved['optimizer']['state'].keys()
    for index, entries in saved['optimizer']['state'].items():
        for key, value in entries.items():
            assert torch.equal(state[index][key], value), (index, key)
    assert int(_subnormal(state[0]['exp_avg']).sum()) == 1
    assert model.scheduler.state_dict()['last_epoch'] == \
        saved['scheduler']['last_epoch'] == 1


def test_a_fresh_model_is_not_rewritten(tmp_path, args, monkeypatch):
    """With no checkpoint there is nothing to clean, so nothing is rewritten."""
    from CSONIC import csonic_model

    calls = []
    monkeypatch.setattr(csonic_model, 'zero_subnormal_weights',
                        lambda net: calls.append(net) or 0)
    args.outdir = str(tmp_path)
    args.ckpt_path = ''

    model = CSONICModel(args, device='cpu')

    assert model.start_step == 0
    assert model.zeroed_subnormal == 0
    assert calls == []


def test_csonic_test_keeps_the_old_helper_name():
    """CSONIC_test._zero_subnormal_weights is the loader's own function."""
    import CSONIC_test
    from CSONIC import csonic_model

    assert CSONIC_test._zero_subnormal_weights is csonic_model.zero_subnormal_weights


@pytest.mark.parametrize('flag', [None, True, False])
def test_csonic_test_load_model_passes_the_switch_to_the_model(monkeypatch, flag):
    """CSONIC_test.load_model hands its switch to CSONICModel.

    With the switch off, it leaves the loaded weights alone as well.
    """
    import CSONIC_test

    built = []

    class RecordingModel:
        def __init__(self, args, device=None, zero_subnormal_weights=True):
            built.append({'args': args, 'device': device,
                          'zero_subnormal_weights': zero_subnormal_weights})
            self.device = device
            self.model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                self.model.weight[0, 0] = POSITIVE_SUBNORMAL

    monkeypatch.setattr(CSONIC_test, 'CSONICModel', RecordingModel)
    kwargs = {} if flag is None else {'zero_subnormal_weights': flag}

    model = CSONIC_test.load_model('some.pth', device='cpu', **kwargs)

    assert len(built) == 1
    assert built[0]['device'] == 'cpu'
    assert built[0]['zero_subnormal_weights'] is (True if flag is None else flag)
    assert built[0]['args'].ckpt_path == 'some.pth'
    assert built[0]['args'].phase == 'test'
    assert not model.model.training
    if flag is False:
        assert int(_subnormal(model.model.weight).sum()) == 1
