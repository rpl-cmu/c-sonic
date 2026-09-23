"""Tests for the sonar dataset: pose algebra, pair lists and one real item."""
import os

import numpy as np
import pytest
import torch

from dataloader.sonardata import SONAR_MEAN, SONAR_STD, SonarData, denormalize_image


def write_pair_files(root, n_pairs=2, n_pose_pairs=None, n_meta_pairs=None):
    """Write the three parallel pair lists under root/logs. Returns the names."""
    n_pose_pairs = n_pairs if n_pose_pairs is None else n_pose_pairs
    n_meta_pairs = n_pairs if n_meta_pairs is None else n_meta_pairs
    logs = os.path.join(str(root), 'logs')
    os.makedirs(logs, exist_ok=True)
    counts = {'pairs.txt': n_pairs, 'pairs_pos.txt': n_pose_pairs,
              'pairs_meta.txt': n_meta_pairs}
    folders = {'pairs.txt': 'images', 'pairs_pos.txt': 'poses',
               'pairs_meta.txt': 'meta'}
    for name, count in counts.items():
        folder = folders[name]
        lines = ['{f}/a{i}.npy {f}/b{i}.npy'.format(f=folder, i=i) for i in range(count)]
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('\n'.join(lines) + '\n')
    return logs


# --------------------------------------------------------------------------
# pose algebra
# --------------------------------------------------------------------------

def test_get_extrinsics():
    """The extrinsic matrix inverts the camera-to-world pose."""
    # roll 30 from holoocean
    pose = np.array(
        [[1, -0, 0, -0.007],
         [-0, 0.866, -0.5, 0.02],
         [0, 0.5, 0.866, -2.68],
         [0, 0, 0, 1]]
    )
    new_pose = SonarData.get_extrinsics(pose)
    np.testing.assert_allclose(pose[:3, :3], new_pose[:3, :3].T, atol=1e-3)
    np.testing.assert_allclose(pose[:3, 3], -new_pose[:3, :3].T @ new_pose[:3, 3],
                               atol=1e-3)


def test_relative_transformation():
    """The relative pose maps a point seen from one frame into the other."""
    pose_origin = np.array(
        [[1., -0., 0., -0.007],
         [-0., 1., -0., 0.02],
         [0., -0., 1., -2.68],
         [0., 0., 0., 1.]]
    )
    pose_x_forward = np.array(
        [[1., -0., 0., 0.993],
         [-0., 1., -0., 0.02],
         [0., -0., 1., -2.68],
         [0., 0., 0., 1.]]
    )
    pose_x_forward_pitch_30 = np.array(
        [[0.866, -0., 0.5, -1.007],
         [-0., 1., -0., 0.02],
         [-0.5, -0., 0.866, -2.68],
         [0., 0., 0., 1.]]
    )
    SonarData.get_extrinsics(pose_origin)
    new_pose_x_forward = SonarData.get_extrinsics(pose_x_forward)
    new_pose_x_forward_pitch_30 = SonarData.get_extrinsics(pose_x_forward_pitch_30)
    new_pose_x_forward_origin = new_pose_x_forward @ np.array([0, 0, 0, 1])
    np.testing.assert_allclose(-pose_x_forward[0, 3], new_pose_x_forward_origin[0])
    x = np.array([1, 2, 3, 1])
    x1 = new_pose_x_forward @ x
    x2 = new_pose_x_forward_pitch_30 @ x
    relative = SonarData.get_relative_pose(new_pose_x_forward_pitch_30, new_pose_x_forward)
    np.testing.assert_allclose(x2, relative @ x1)


# --------------------------------------------------------------------------
# pair lists
# --------------------------------------------------------------------------

def test_read_pairs_relative_paths(tmp_path, args):
    """Relative pair file names are read under the dataset root."""
    write_pair_files(tmp_path, n_pairs=2)
    args.phase = 'train'
    args.datadir = str(tmp_path)
    dataset = SonarData(args)

    assert len(dataset) == 2
    for path in dataset.imf1s + dataset.imf2s + dataset.posf1s + dataset.metaf1s:
        assert path.startswith(str(tmp_path))
    assert sorted(os.path.basename(p) for p in dataset.imf1s) == ['a0.npy', 'a1.npy']
    assert sorted(os.path.basename(p) for p in dataset.posf2s) == ['b0.npy', 'b1.npy']


def test_read_pairs_absolute_paths(tmp_path, args):
    """An absolute pair file name is used as given, whatever the root is."""
    logs = write_pair_files(tmp_path, n_pairs=2)
    other_root = tmp_path / 'root'
    other_root.mkdir()
    args.phase = 'train'
    args.datadir = str(other_root)
    args.pairs_file = os.path.join(logs, 'pairs.txt')
    args.pairs_pos_file = os.path.join(logs, 'pairs_pos.txt')
    args.pairs_meta_file = os.path.join(logs, 'pairs_meta.txt')
    dataset = SonarData(args)

    assert len(dataset) == 2
    # The entries inside the list stay relative to the dataset root.
    assert all(p.startswith(str(other_root)) for p in dataset.imf1s)


def test_read_pairs_val_phase_uses_val_files(tmp_path, args):
    """The val phase reads the val pair lists from the val dataset root."""
    val_root = tmp_path / 'val'
    logs = os.path.join(str(val_root), 'logs')
    os.makedirs(logs)
    for name in ('pairs_val.txt', 'pairs_pos_val.txt', 'pairs_meta_val.txt'):
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('x/a0.npy x/b0.npy\nx/a1.npy x/b1.npy\nx/a2.npy x/b2.npy\n')
    args.phase = 'val'
    args.datadir = str(tmp_path)
    args.val_data_dir = str(val_root)
    dataset = SonarData(args)

    assert len(dataset) == 3
    assert all(p.startswith(str(val_root)) for p in dataset.imf1s)


def test_read_pairs_val_phase_falls_back_to_datadir(tmp_path, args):
    """With no --val_data_dir the val phase reads from --datadir."""
    logs = os.path.join(str(tmp_path), 'logs')
    os.makedirs(logs)
    for name in ('pairs_val.txt', 'pairs_pos_val.txt', 'pairs_meta_val.txt'):
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('x/a0.npy x/b0.npy\n')
    args.phase = 'val'
    args.datadir = str(tmp_path)
    args.val_data_dir = ''
    dataset = SonarData(args)

    assert len(dataset) == 1
    assert dataset.root == str(tmp_path)


def test_read_pairs_missing_file_raises(tmp_path, args):
    """A missing pair list is an error that names the path it looked for."""
    args.phase = 'train'
    args.datadir = str(tmp_path)
    expected = os.path.join(str(tmp_path), 'logs/pairs.txt')
    with pytest.raises(FileNotFoundError) as excinfo:
        SonarData(args)
    assert expected in str(excinfo.value)


def test_read_pairs_length_mismatch_raises(tmp_path, args):
    """Pair lists of different lengths are an error, not a silent truncation."""
    write_pair_files(tmp_path, n_pairs=3, n_pose_pairs=2)
    args.phase = 'train'
    args.datadir = str(tmp_path)
    with pytest.raises(ValueError):
        SonarData(args)


def test_read_pairs_splits_on_any_whitespace(tmp_path, args):
    """Lines separated by a tab or by several spaces still parse."""
    logs = os.path.join(str(tmp_path), 'logs')
    os.makedirs(logs)
    for name in ('pairs.txt', 'pairs_pos.txt', 'pairs_meta.txt'):
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('x/a0.npy\tx/b0.npy\nx/a1.npy   x/b1.npy\n')
    args.phase = 'train'
    args.datadir = str(tmp_path)
    dataset = SonarData(args)

    assert len(dataset) == 2
    assert sorted(os.path.basename(p) for p in dataset.imf2s) == ['b0.npy', 'b1.npy']


def test_read_pairs_blank_line_in_middle_raises(tmp_path, args):
    """A blank line is an error, not something to skip.

    Skipping it silently would shift the pairing: a blank line 2 in the image
    list and a blank line 3 in the pose list keep both lengths equal while
    pairing the wrong files.
    """
    logs = os.path.join(str(tmp_path), 'logs')
    os.makedirs(logs)
    with open(os.path.join(logs, 'pairs.txt'), 'w') as handle:
        handle.write('x/a0.npy x/b0.npy\n\nx/a1.npy x/b1.npy\n')
    for name in ('pairs_pos.txt', 'pairs_meta.txt'):
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('x/a0.npy x/b0.npy\nx/a1.npy x/b1.npy\n\n')
    args.phase = 'train'
    args.datadir = str(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        SonarData(args)
    message = str(excinfo.value)
    assert os.path.join(logs, 'pairs.txt') in message
    assert ':2:' in message


@pytest.mark.parametrize('bad_line', ['x/only_one.npy', 'x/a.npy x/b.npy x/c.npy'])
def test_read_pairs_malformed_line_raises(tmp_path, args, bad_line):
    """A line without exactly two paths names the file and the line number."""
    logs = os.path.join(str(tmp_path), 'logs')
    os.makedirs(logs)
    with open(os.path.join(logs, 'pairs.txt'), 'w') as handle:
        handle.write('x/a0.npy x/b0.npy\nx/a1.npy x/b1.npy\n{}\n'.format(bad_line))
    for name in ('pairs_pos.txt', 'pairs_meta.txt'):
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('x/a0.npy x/b0.npy\nx/a1.npy x/b1.npy\nx/a2.npy x/b2.npy\n')
    args.phase = 'train'
    args.datadir = str(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        SonarData(args)
    message = str(excinfo.value)
    assert os.path.join(logs, 'pairs.txt') in message
    assert ':3:' in message
    assert bad_line in message


def test_read_pairs_trailing_blank_lines_are_fine(tmp_path, args):
    """A final newline, or several, is not an error."""
    logs = os.path.join(str(tmp_path), 'logs')
    os.makedirs(logs)
    for name, tail in (('pairs.txt', '\n'), ('pairs_pos.txt', '\n\n\n'),
                       ('pairs_meta.txt', '\n  \n')):
        with open(os.path.join(logs, name), 'w') as handle:
            handle.write('x/a0.npy x/b0.npy\nx/a1.npy x/b1.npy' + tail)
    args.phase = 'train'
    args.datadir = str(tmp_path)
    dataset = SonarData(args)

    assert len(dataset) == 2


def test_datadir_required(args):
    """An empty dataset root is an error with a message that names the flag."""
    args.phase = 'train'
    args.datadir = ''
    args.val_data_dir = ''
    with pytest.raises(ValueError) as excinfo:
        SonarData(args)
    assert '--datadir' in str(excinfo.value)


# --------------------------------------------------------------------------
# image normalization
# --------------------------------------------------------------------------

def test_denormalize_image_roundtrip(tmp_path, args):
    """denormalize_image undoes the transform the dataset applies."""
    write_pair_files(tmp_path, n_pairs=2)
    args.phase = 'train'
    args.datadir = str(tmp_path)
    dataset = SonarData(args)

    rng = np.random.RandomState(0)
    image = rng.rand(32, 32).astype(np.float32)
    tensor = dataset.transform(image)
    # The transform really changes the values, so the roundtrip is not trivial.
    assert not torch.allclose(tensor, torch.from_numpy(image).unsqueeze(0), atol=1e-3)
    torch.testing.assert_close(denormalize_image(tensor)[0], torch.from_numpy(image),
                               atol=1e-6, rtol=0)


def test_normalization_constants_used_by_transform(tmp_path, args):
    """The dataset normalizes with the module constants, not its own numbers."""
    write_pair_files(tmp_path, n_pairs=2)
    args.phase = 'train'
    args.datadir = str(tmp_path)
    dataset = SonarData(args)

    image = np.full((8, 8), 0.5, dtype=np.float32)
    tensor = dataset.transform(image)
    expected = (0.5 - SONAR_MEAN) / SONAR_STD
    torch.testing.assert_close(tensor, torch.full((1, 8, 8), expected), atol=1e-5,
                               rtol=0)


# --------------------------------------------------------------------------
# a real item
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_getitem_tiny_dataset(tiny_dataset_dir, args):
    """One item of the real dataset carries the keys the model reads."""
    args.phase = 'train'
    args.datadir = tiny_dataset_dir
    args.num_pts = 50
    dataset = SonarData(args)
    assert len(dataset) == 21

    item = dataset[0]
    expected_keys = {
        'im1', 'im2', 'pose', 'T1', 'T2', 'coord1',
        'im1_width', 'im1_height', 'im2_width', 'im2_height',
        'im1_r_min', 'im1_r_max', 'im2_r_min', 'im2_r_max',
        'im1_elev', 'im1_azi', 'im2_elev', 'im2_azi',
    }
    assert set(item.keys()) == expected_keys
    assert item['im1'].shape == (1, 512, 512)
    assert item['im2'].shape == (1, 512, 512)
    assert item['coord1'].shape == (args.num_pts, 2)
    assert item['T1'].shape == (4, 4)
    assert item['T2'].shape == (4, 4)
    assert item['pose'].shape == (4, 4)

    meta = np.load(dataset.metaf1s[0], allow_pickle=True).item()
    assert item['im1_width'] == meta['width']
    assert item['im1_height'] == meta['height']
    assert item['im1_r_min'] == meta['r_min']
    assert item['im1_r_max'] == meta['r_max']
    assert item['im1_elev'] == meta['elev']
    assert item['im1_azi'] == meta['azi']

    # The query points sit inside the image.
    coord = item['coord1'].numpy()
    assert coord[:, 0].min() >= 0 and coord[:, 0].max() <= meta['width']
    assert coord[:, 1].min() >= 0 and coord[:, 1].max() <= meta['height']
