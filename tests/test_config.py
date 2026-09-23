"""Tests for the command line configuration.

These tests call ``config.get_args`` directly rather than using the ``args``
fixture. The fixture adds legacy attributes so the pre-cleanup modules can still
be built, and those attributes would hide a removed argument coming back.
"""
import config

# All twenty arguments the cleanup removed. None of them may come back.
REMOVED_ARGS = [
    'rootdir', 'logdir',
    'range_max', 'range_end', 'range_start', 'horizontal_fov', 'vertical_fov',
    'curriculum', 'curr_diff',
    'attention', 'clip',
    'prune_kp', 'train_kp', 'akaze_pts',
    'std', 'w_std', 'th_cycle', 'th_epipolar',
    'extract_img_dir', 'extract_out_dir',
]

# The hyper-parameters the released checkpoint was trained with.
RELEASED_DEFAULTS = {
    'backbone': 'resnet34',
    'pretrained': 1,
    'coarse_feat_dim': 64,
    'fine_feat_dim': 64,
    'cross_attention': 1,
    'num_pts': 50,
    'num_samples': 100,
    'akaze_superpoint_pts': 20,
    'batch_size': 14,
    'lr': 1e-4,
    'lrate_decay_steps': 30000,
    'lrate_decay_factor': 0.5,
    'weight_decay': 1e-4,
    'w_epipolar_coarse': 9,
    'w_epipolar_fine': 7,
    'w_cycle_coarse': 1,
    'w_cycle_fine': 1,
    'window_size': 0.125,
    'use_nn': 1,
    'prob_from': 'correlation',
    'log_scalar_interval': 20,
    'log_img_interval': 2000,
    'save_interval': 20000,
    'n_val_iters': 4303,
    'num_kpts_shown': 25,
}

# The pair list arguments and the paths they default to.
PAIR_FILE_DEFAULTS = {
    'pairs_file': 'logs/pairs.txt',
    'pairs_pos_file': 'logs/pairs_pos.txt',
    'pairs_meta_file': 'logs/pairs_meta.txt',
    'val_pairs_file': 'logs/pairs_val.txt',
    'val_pairs_pos_file': 'logs/pairs_pos_val.txt',
    'val_pairs_meta_file': 'logs/pairs_meta_val.txt',
}


def test_get_args_accepts_empty_argv():
    """get_args takes an argv list, so tests and notebooks need no sys.argv."""
    args = config.get_args([])
    assert args.phase == 'train'


def test_get_args_reads_argv():
    """A value on the command line overrides the default."""
    assert config.get_args(['--batch_size', '3']).batch_size == 3


def test_no_user_paths_in_defaults():
    """No default may point at a developer machine."""
    values = vars(config.get_args([]))
    strings = {k: v for k, v in values.items() if isinstance(v, str)}
    # Guard against a vacuous pass: there really are string defaults to check.
    assert len(strings) >= 5
    offenders = {k: v for k, v in strings.items()
                 if '/home/' in v or '/media/' in v}
    assert offenders == {}


def test_removed_args_absent():
    """Every removed argument is gone from the parsed namespace."""
    args = config.get_args([])
    # Guard against a vacuous pass: the list really is the twenty deleted options.
    assert len(REMOVED_ARGS) == 20 and len(set(REMOVED_ARGS)) == 20
    present = [name for name in REMOVED_ARGS if hasattr(args, name)]
    assert present == []


def test_released_defaults():
    """The defaults reproduce the released model's hyper-parameters."""
    args = config.get_args([])
    actual = {name: getattr(args, name) for name in RELEASED_DEFAULTS}
    assert actual == RELEASED_DEFAULTS


def test_data_dir_defaults_are_empty():
    """The dataset directories have no default, so a run must name one."""
    args = config.get_args([])
    assert args.datadir == ''
    assert args.val_data_dir == ''


def test_pair_file_defaults():
    """The pair list arguments exist with the documented relative defaults."""
    args = config.get_args([])
    actual = {name: getattr(args, name) for name in PAIR_FILE_DEFAULTS}
    assert actual == PAIR_FILE_DEFAULTS


def test_superpoint_and_wandb_defaults():
    """The SuperPoint weights and the wandb switches have release defaults."""
    args = config.get_args([])
    assert args.superpoint_weights == 'pretrained/superpoint_v1.pth'
    assert args.wandb == 0
    assert args.wandb_project == 'c-sonic'


def test_outdir_and_exp_name_defaults():
    """Checkpoints land in ./checkpoints/ under the released experiment name."""
    args = config.get_args([])
    assert args.outdir == './checkpoints/'
    assert args.exp_name == 'csonic_resnet34_64dim'
    assert args.ckpt_path == ''
    assert args.n_iters == 500000
