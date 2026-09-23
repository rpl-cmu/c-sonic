"""Tests for extract_sample.py, the sample extractor.

The tests build small synthetic zip files in a temporary directory. They read
no data from the real dataset. The members are real .npy files, so the tests
can compare the written files with the archive byte for byte.
"""
import hashlib
import io
import json
import os
import zipfile

import numpy as np
import pytest

import extract_sample
import make_pairs

# Three synthetic logs of one scene. The names follow the release convention,
# so the tool picks T_BPL_1 as the anchor on its own.
LOGS = ['T_BPH_1', 'T_BPL_1', 'T_BPL_1_LR']
ANCHOR = 'T_BPL_1'
# A second scene, for the tests of --append. It holds two logs and a smaller
# frame gap, as the second scene of the release sample does.
LOGS_B = ['U_BPH_1', 'U_BPL_1']
ANCHOR_B = 'U_BPL_1'
FRAMES_B = [4, 8, 12]
MAX_GAP_B = 4
ALL_LOGS = LOGS + LOGS_B
N_SONARS = 2
N_FRAMES = 25
FRAMES = [10, 15, 20]
META = {'width': 4, 'height': 4, 'r_min': 0.1, 'r_max': 10.0, 'elev': 12.0,
        'azi': 60.0}
KIND_PARTS = (('Sonar', 'S'), ('Pose', 'P'), ('Metadata', 'M'))

# The 16 lines that the tool must write, in this order: the cross-log pairs of
# every other log, in the order of the scene, then the same-log pairs. In a
# pair of the anchor and the high-frequency log T_BPH_1, the high-frequency
# image comes first. The other pairs start with the anchor image.
EXPECTED_PAIRS = [
    'logs/T_BPH_1/Sonar_0/S_10.npy logs/T_BPL_1/Sonar_0/S_10.npy',
    'logs/T_BPH_1/Sonar_0/S_15.npy logs/T_BPL_1/Sonar_0/S_10.npy',
    'logs/T_BPH_1/Sonar_0/S_10.npy logs/T_BPL_1/Sonar_0/S_15.npy',
    'logs/T_BPH_1/Sonar_0/S_15.npy logs/T_BPL_1/Sonar_0/S_15.npy',
    'logs/T_BPH_1/Sonar_0/S_20.npy logs/T_BPL_1/Sonar_0/S_15.npy',
    'logs/T_BPH_1/Sonar_0/S_15.npy logs/T_BPL_1/Sonar_0/S_20.npy',
    'logs/T_BPH_1/Sonar_0/S_20.npy logs/T_BPL_1/Sonar_0/S_20.npy',
    'logs/T_BPL_1/Sonar_0/S_10.npy logs/T_BPL_1_LR/Sonar_0/S_10.npy',
    'logs/T_BPL_1/Sonar_0/S_10.npy logs/T_BPL_1_LR/Sonar_0/S_15.npy',
    'logs/T_BPL_1/Sonar_0/S_15.npy logs/T_BPL_1_LR/Sonar_0/S_10.npy',
    'logs/T_BPL_1/Sonar_0/S_15.npy logs/T_BPL_1_LR/Sonar_0/S_15.npy',
    'logs/T_BPL_1/Sonar_0/S_15.npy logs/T_BPL_1_LR/Sonar_0/S_20.npy',
    'logs/T_BPL_1/Sonar_0/S_20.npy logs/T_BPL_1_LR/Sonar_0/S_15.npy',
    'logs/T_BPL_1/Sonar_0/S_20.npy logs/T_BPL_1_LR/Sonar_0/S_20.npy',
    'logs/T_BPL_1/Sonar_0/S_10.npy logs/T_BPL_1/Sonar_0/S_15.npy',
    'logs/T_BPL_1/Sonar_0/S_15.npy logs/T_BPL_1/Sonar_0/S_20.npy',
]

# With a largest gap of zero only the pairs of one position are left, and no
# same-log pair can survive.
EXPECTED_GAP_ZERO = [
    'logs/T_BPH_1/Sonar_0/S_10.npy logs/T_BPL_1/Sonar_0/S_10.npy',
    'logs/T_BPH_1/Sonar_0/S_15.npy logs/T_BPL_1/Sonar_0/S_15.npy',
    'logs/T_BPH_1/Sonar_0/S_20.npy logs/T_BPL_1/Sonar_0/S_20.npy',
    'logs/T_BPL_1/Sonar_0/S_10.npy logs/T_BPL_1_LR/Sonar_0/S_10.npy',
    'logs/T_BPL_1/Sonar_0/S_15.npy logs/T_BPL_1_LR/Sonar_0/S_15.npy',
    'logs/T_BPL_1/Sonar_0/S_20.npy logs/T_BPL_1_LR/Sonar_0/S_20.npy',
]

# The nine lines that the second scene adds: seven cross-log pairs with a gap
# of zero or four frames, the high-frequency image first, then two same-log
# pairs.
EXPECTED_APPEND = [
    'logs/U_BPH_1/Sonar_0/S_4.npy logs/U_BPL_1/Sonar_0/S_4.npy',
    'logs/U_BPH_1/Sonar_0/S_8.npy logs/U_BPL_1/Sonar_0/S_4.npy',
    'logs/U_BPH_1/Sonar_0/S_4.npy logs/U_BPL_1/Sonar_0/S_8.npy',
    'logs/U_BPH_1/Sonar_0/S_8.npy logs/U_BPL_1/Sonar_0/S_8.npy',
    'logs/U_BPH_1/Sonar_0/S_12.npy logs/U_BPL_1/Sonar_0/S_8.npy',
    'logs/U_BPH_1/Sonar_0/S_8.npy logs/U_BPL_1/Sonar_0/S_12.npy',
    'logs/U_BPH_1/Sonar_0/S_12.npy logs/U_BPL_1/Sonar_0/S_12.npy',
    'logs/U_BPL_1/Sonar_0/S_4.npy logs/U_BPL_1/Sonar_0/S_8.npy',
    'logs/U_BPL_1/Sonar_0/S_8.npy logs/U_BPL_1/Sonar_0/S_12.npy',
]
EXPECTED_MERGED = EXPECTED_PAIRS + EXPECTED_APPEND

# The nine images the 16 pairs name: three frames of each of the three logs.
EXPECTED_IMAGES = ['logs/{}/Sonar_0/S_{}.npy'.format(log, frame)
                   for log in (ANCHOR, 'T_BPH_1', 'T_BPL_1_LR')
                   for frame in FRAMES]
LIST_NAMES = make_pairs.MERGED_NAMES['train'] + make_pairs.MERGED_NAMES['val']


# --------------------------------------------------------------------------
# the synthetic archives
# --------------------------------------------------------------------------

def npy_bytes(array):
    """Return the bytes of one .npy file."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=True)
    return buffer.getvalue()


def image_array(log, sonar, frame):
    """A 4x4 image. Every (log, sonar, frame) holds its own values."""
    base = ALL_LOGS.index(log) * 10000 + sonar * 1000 + frame
    return np.arange(16, dtype=np.float64).reshape(4, 4) + float(base)


def pose_array(frame):
    """A 4x4 pose. The frame number is the position on the first axis."""
    pose = np.eye(4)
    pose[0, 3] = float(frame)
    return pose


def members(log):
    """Yield (member name, bytes) for every file of one log."""
    for sonar in range(N_SONARS):
        for frame in range(N_FRAMES):
            yield ('{}/Sonar_{}/S_{}.npy'.format(log, sonar, frame),
                   npy_bytes(image_array(log, sonar, frame)))
            yield ('{}/Pose_{}/P_{}.npy'.format(log, sonar, frame),
                   npy_bytes(pose_array(frame)))
            yield ('{}/Metadata_{}/M_{}.npy'.format(log, sonar, frame),
                   npy_bytes(np.array(META, dtype=object)))


def make_zip(folder, log):
    """Write one log as a zip in the dataset layout. Returns the path."""
    path = os.path.join(str(folder), log + '.zip')
    with zipfile.ZipFile(path, 'w') as handle:
        for name, data in members(log):
            handle.writestr(name, data)
    return path


def run(zips, out, extra=(), frames=('10', '15', '20')):
    """Run the tool on the synthetic logs. Returns the exit code."""
    argv = (['--zips'] + list(zips) + ['--logs'] + LOGS
            + ['--out', str(out), '--sonar', '0', '--frames'] + list(frames)
            + list(extra))
    return extract_sample.main(argv)


def run_b(zips_b, out, extra=(), frames=FRAMES_B):
    """Run the tool on the second scene. Returns the exit code."""
    argv = (['--zips'] + list(zips_b) + ['--logs'] + LOGS_B
            + ['--out', str(out), '--sonar', '0', '--frames']
            + [str(frame) for frame in frames]
            + ['--max-gap', str(MAX_GAP_B)] + list(extra))
    return extract_sample.main(argv)


def is_empty(path):
    """Tell whether the folder is absent or holds nothing."""
    return not os.path.exists(str(path)) or os.listdir(str(path)) == []


def npy_files(sample):
    """Return the .npy files of a sample, sorted."""
    found = []
    for base, _, names in os.walk(os.path.join(str(sample), 'logs')):
        found.extend(os.path.join(base, name) for name in names
                     if name.endswith('.npy'))
    return sorted(found)


def list_bytes(sample):
    """Return the bytes of the six pair lists."""
    return {name: read_file(str(sample), 'logs', name) for name in LIST_NAMES}


def snapshot(sample):
    """Return the md5 sum and the write time of every file of a sample.

    Two equal snapshots mean that no file was added, removed, rewritten or
    touched.
    """
    root = str(sample)
    state = {}
    for base, _, names in os.walk(root):
        for name in names:
            path = os.path.join(base, name)
            state[os.path.relpath(path, root)] = (
                hashlib.md5(read_file(path)).hexdigest(),
                os.stat(path).st_mtime_ns)
    return state


def read_list(sample, name):
    """Read one pair list. The file must end with a line break."""
    with open(os.path.join(sample, 'logs', name)) as handle:
        text = handle.read()
    assert text.endswith('\n'), '{} does not end with a line break'.format(name)
    return text.splitlines()


def read_file(*parts):
    """Return the bytes of one file."""
    with open(os.path.join(*parts), 'rb') as handle:
        return handle.read()


@pytest.fixture(scope='module')
def zips(tmp_path_factory):
    """One zip file per synthetic log."""
    folder = tmp_path_factory.mktemp('zips')
    return [make_zip(folder, log) for log in LOGS]


@pytest.fixture(scope='module')
def zips_b(tmp_path_factory):
    """One zip file per log of the second scene."""
    folder = tmp_path_factory.mktemp('zips_b')
    return [make_zip(folder, log) for log in LOGS_B]


@pytest.fixture(scope='module')
def sample(tmp_path_factory, zips):
    """A sample built with the standard options, for the read-only tests."""
    out = tmp_path_factory.mktemp('sample') / 'csonic'
    assert run(zips, out) == 0
    return str(out)


@pytest.fixture(scope='module')
def appended(tmp_path_factory, zips, zips_b):
    """A sample of the first scene with the second scene added to it."""
    out = tmp_path_factory.mktemp('appended') / 'csonic'
    assert run(zips, out) == 0
    assert run_b(zips_b, out, extra=['--append']) == 0
    return str(out)


# --------------------------------------------------------------------------
# the pairs
# --------------------------------------------------------------------------

def test_pair_list_holds_the_sixteen_expected_lines(sample):
    """The image list is the 16 pairs, in the order of the design."""
    lines = read_list(sample, 'pairs.txt')
    assert lines == EXPECTED_PAIRS
    assert len(lines) == 16


def test_max_gap_zero_drops_the_moved_pairs(tmp_path, zips):
    """A largest gap of zero leaves the six pairs of one position only."""
    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--max-gap', '0']) == 0
    assert read_list(str(out), 'pairs.txt') == EXPECTED_GAP_ZERO


# --------------------------------------------------------------------------
# the order inside a pair
# --------------------------------------------------------------------------

def log_of(image):
    """Return the log of one image path."""
    return make_pairs.parse_image_path(image)[0]


def anchor_first_pairs(logs, anchor, sonar, frames, max_gap):
    """The pairs of the earlier rule: the anchor image first in every pair.

    The test builds them here, apart from the tool, so that it can compare the
    set of pairs of the new rule with the set of the earlier rule.
    """
    pairs = []
    for other in logs:
        if other == anchor:
            continue
        for first in frames:
            for second in frames:
                if abs(first - second) <= max_gap:
                    pairs.append((make_pairs.image_path(anchor, sonar, first),
                                  make_pairs.image_path(other, sonar, second)))
    for first in frames:
        for second in frames:
            if 0 < second - first <= max_gap:
                pairs.append((make_pairs.image_path(anchor, sonar, first),
                              make_pairs.image_path(anchor, sonar, second)))
    return pairs


def test_is_high_frequency_reads_the_log_name():
    """A log is high frequency when its name holds '_BPH_'."""
    for log in ('T_BPH_1', 'ELC2_BPH_1', 'ELC1_BPH_1', 'U_BPH_2'):
        assert extract_sample.is_high_frequency(log), log
    for log in ('T_BPL_1', 'T_BPL_1_LR', 'ELC2_BPL_1', 'RW_BPL_1_LR', 'BPH',
                'T_BPH'):
        assert not extract_sample.is_high_frequency(log), log


def test_a_low_frequency_anchor_puts_the_high_frequency_image_first():
    """In every pair of T_BPL_1 and T_BPH_1, the T_BPH_1 image comes first."""
    pairs = extract_sample.build_pairs(LOGS, ANCHOR, 0, FRAMES, 5)
    with_bph = [pair for pair in pairs
                if 'T_BPH_1' in (log_of(pair[0]), log_of(pair[1]))]
    assert len(with_bph) == 7
    for first, second in with_bph:
        assert log_of(first) == 'T_BPH_1', (first, second)
        assert log_of(second) == ANCHOR, (first, second)


def test_the_long_range_pairs_keep_the_anchor_first():
    """A pair without a high-frequency image starts with the anchor image."""
    pairs = extract_sample.build_pairs(LOGS, ANCHOR, 0, FRAMES, 5)
    with_lr = [pair for pair in pairs
               if 'T_BPL_1_LR' in (log_of(pair[0]), log_of(pair[1]))]
    assert len(with_lr) == 7
    for first, second in with_lr:
        assert log_of(first) == ANCHOR, (first, second)
        assert log_of(second) == 'T_BPL_1_LR', (first, second)


def test_the_same_log_pairs_are_unchanged():
    """The same-log pairs come last, the earlier frame first."""
    pairs = extract_sample.build_pairs(LOGS, ANCHOR, 0, FRAMES, 5)
    assert pairs[-2:] == [
        (make_pairs.image_path(ANCHOR, 0, 10), make_pairs.image_path(ANCHOR, 0, 15)),
        (make_pairs.image_path(ANCHOR, 0, 15), make_pairs.image_path(ANCHOR, 0, 20)),
    ]
    assert pairs[-2:] == anchor_first_pairs(LOGS, ANCHOR, 0, FRAMES, 5)[-2:]


@pytest.mark.parametrize('max_gap', [0, 4, 5, 10])
@pytest.mark.parametrize('logs, anchor, frames', [
    (LOGS, ANCHOR, FRAMES),
    (LOGS_B, ANCHOR_B, FRAMES_B),
    (make_pairs.DEFAULT_SCENES['ELC2'], 'ELC2_BPL_1', [1000, 1005, 1010]),
    (make_pairs.DEFAULT_SCENES['ELC1'], 'ELC1_BPL_1', [1200, 1204, 1208]),
    (make_pairs.DEFAULT_SCENES['RW'], 'RW_BPL_1', [1, 2, 3]),
    (LOGS, 'T_BPH_1', FRAMES),
])
def test_the_set_of_pairs_is_the_set_of_the_earlier_rule(logs, anchor, frames,
                                                         max_gap):
    """Only the order inside a pair changes: the pairs are the same pairs.

    Line n of the new list is line n of the earlier list, with its two images
    swapped exactly when the pair holds one high-frequency image.
    """
    new = extract_sample.build_pairs(logs, anchor, 0, frames, max_gap)
    old = anchor_first_pairs(logs, anchor, 0, frames, max_gap)
    assert len(new) == len(old) == len(set(new))
    assert set(frozenset(pair) for pair in new) \
        == set(frozenset(pair) for pair in old)
    for (new_first, new_second), (old_first, old_second) in zip(new, old):
        high = [extract_sample.is_high_frequency(log_of(image))
                for image in (old_first, old_second)]
        if high == [False, True]:
            assert (new_first, new_second) == (old_second, old_first)
        else:
            assert (new_first, new_second) == (old_first, old_second)


def test_an_explicit_high_frequency_anchor_comes_first(tmp_path, zips):
    """--anchor T_BPH_1 gives the high-frequency image first in every pair."""
    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--anchor', 'T_BPH_1']) == 0
    lines = read_list(str(out), 'pairs.txt')
    assert len(lines) == 16
    for line in lines:
        first, second = line.split()
        assert log_of(first) == 'T_BPH_1', line
    assert [log_of(line.split()[1]) for line in lines] \
        == ['T_BPL_1'] * 7 + ['T_BPL_1_LR'] * 7 + ['T_BPH_1'] * 2


def test_two_high_frequency_logs_keep_the_anchor_first():
    """A pair of two high-frequency images starts with the anchor image."""
    logs = ['V_BPH_1', 'V_BPH_2', 'V_BPL_1']
    pairs = extract_sample.build_pairs(logs, 'V_BPH_1', 0, [1, 2], 1)
    assert pairs == anchor_first_pairs(logs, 'V_BPH_1', 0, [1, 2], 1)
    assert all(log_of(first) == 'V_BPH_1' for first, _ in pairs)
    # With a low-frequency anchor both high-frequency logs come first.
    pairs = extract_sample.build_pairs(logs, 'V_BPL_1', 0, [1, 2], 1)
    # Two frames one frame apart give four cross-log pairs for each other log.
    cross = [pair for pair in pairs if log_of(pair[0]) != log_of(pair[1])]
    assert len(cross) == 8
    assert all(extract_sample.is_high_frequency(log_of(first))
               and log_of(second) == 'V_BPL_1' for first, second in cross)


# --------------------------------------------------------------------------
# the six lists
# --------------------------------------------------------------------------

def test_the_six_lists_are_parallel_and_derived(sample):
    """The pose and metadata lists follow the image list, line for line."""
    for name in LIST_NAMES:
        assert os.path.isfile(os.path.join(sample, 'logs', name)), name
    for split in ('train', 'val'):
        image_name, pose_name, meta_name = make_pairs.MERGED_NAMES[split]
        images = read_list(sample, image_name)
        assert len(images) == 16
        for name, kind in ((pose_name, 'pose'), (meta_name, 'meta')):
            lines = read_list(sample, name)
            assert len(lines) == len(images)
            assert lines == make_pairs.derive_lines(images, kind)


def test_the_training_lists_and_the_validation_lists_are_the_same(sample):
    """The notebook reads the validation lists, a training run the others."""
    for train_name, val_name in zip(make_pairs.MERGED_NAMES['train'],
                                    make_pairs.MERGED_NAMES['val']):
        train = read_file(sample, 'logs', train_name)
        assert train, train_name
        assert train == read_file(sample, 'logs', val_name)


# --------------------------------------------------------------------------
# the extracted files
# --------------------------------------------------------------------------

def test_every_written_file_equals_the_zip_member(sample, zips):
    """The tool copies the members byte for byte."""
    by_log = {os.path.splitext(os.path.basename(path))[0]: path for path in zips}
    checked = 0
    for log in LOGS:
        with zipfile.ZipFile(by_log[log]) as handle:
            for frame in FRAMES:
                for folder, prefix in KIND_PARTS:
                    name = '{}/{}_0/{}_{}.npy'.format(log, folder, prefix, frame)
                    written = read_file(sample, 'logs', *name.split('/'))
                    assert written == handle.read(name), name
                    checked += 1
    assert checked == 27


def test_only_the_named_frames_are_written(sample):
    """The sample holds the nine images and nothing else."""
    found = []
    for base, _, names in os.walk(os.path.join(sample, 'logs')):
        found.extend(os.path.join(base, name) for name in names
                     if name.endswith('.npy'))
    assert len(found) == 27
    images = sorted(path for path in found if os.sep + 'Sonar_0' + os.sep in path)
    assert [os.path.relpath(path, sample).replace(os.sep, '/') for path in images] \
        == sorted(EXPECTED_IMAGES)


def test_a_missing_frame_raises_and_writes_nothing(tmp_path, zips):
    """The tool checks the whole plan before it writes the first file."""
    out = tmp_path / 'sample'
    with pytest.raises(ValueError) as error:
        run(zips, out, frames=('10', '999'))
    assert 'S_999.npy' in str(error.value)
    assert is_empty(out)


def test_dry_run_writes_nothing(tmp_path, zips):
    """A dry run prints the plan only."""
    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--dry-run']) == 0
    assert is_empty(out)


# --------------------------------------------------------------------------
# the previews
# --------------------------------------------------------------------------

def test_preview_writes_one_png_per_image(tmp_path, zips):
    """Every image of the sample gets a small PNG."""
    from PIL import Image

    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--preview']) == 0
    expected = sorted('{}_S0_F{}.png'.format(log, frame)
                      for log in LOGS for frame in FRAMES)
    assert len(expected) == 9
    folder = out / 'preview'
    assert sorted(os.listdir(str(folder))) == expected
    for name in expected:
        with Image.open(str(folder / name)) as handle:
            assert handle.size == (4, 4)
            assert handle.mode == 'L'


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

def test_manifest_describes_the_sample(sample, zips):
    """The manifest holds the plan, and its sums match the written files."""
    with open(os.path.join(sample, 'sample_manifest.json')) as handle:
        manifest = json.load(handle)

    assert manifest['logs'] == LOGS
    assert manifest['anchor'] == ANCHOR
    assert manifest['scene'] is None
    assert manifest['sonar'] == 0
    assert manifest['frames'] == FRAMES
    assert manifest['max_gap'] == 5
    assert [' '.join(pair) for pair in manifest['pairs']] == EXPECTED_PAIRS
    assert manifest['source_zips'] == [log + '.zip' for log in LOGS]
    assert all(os.sep not in name for name in manifest['source_zips'])

    assert manifest['files'] == 27
    assert len(manifest['md5']) == 27
    total = 0
    for path, digest in manifest['md5'].items():
        data = read_file(sample, *path.split('/'))
        assert hashlib.md5(data).hexdigest() == digest, path
        total += len(data)
    assert manifest['total_bytes'] == total > 0


# --------------------------------------------------------------------------
# the dataloader
# --------------------------------------------------------------------------

def test_the_sample_loads_through_the_dataloader(sample):
    """The release dataloader reads the sample with the default options."""
    import config
    from dataloader.sonardata import SonarData

    args = config.get_args([])
    args.datadir = sample
    args.phase = 'test'
    dataset = SonarData(args)
    assert len(dataset) == 16
    names = (dataset.imf1s + dataset.imf2s + dataset.posf1s + dataset.posf2s
             + dataset.metaf1s + dataset.metaf2s)
    assert len(names) == 96
    for name in names:
        assert os.path.isfile(name), name


# --------------------------------------------------------------------------
# --append: a second scene in the same folder
# --------------------------------------------------------------------------

def test_append_keeps_the_old_lines_first(appended):
    """The new pairs come after the pairs of the first run, in order."""
    lines = read_list(appended, 'pairs.txt')
    assert lines[:16] == EXPECTED_PAIRS
    assert lines[16:] == EXPECTED_APPEND
    assert lines == EXPECTED_MERGED
    assert len(lines) == 25


def test_append_keeps_the_six_lists_parallel(appended):
    """Every list holds the 25 pairs, and the three kinds stay in step."""
    for split in ('train', 'val'):
        image_name, pose_name, meta_name = make_pairs.MERGED_NAMES[split]
        images = read_list(appended, image_name)
        assert images == EXPECTED_MERGED
        for name, kind in ((pose_name, 'pose'), (meta_name, 'meta')):
            lines = read_list(appended, name)
            assert len(lines) == 25
            assert lines == make_pairs.derive_lines(images, kind)
    for train_name, val_name in zip(make_pairs.MERGED_NAMES['train'],
                                    make_pairs.MERGED_NAMES['val']):
        assert read_file(appended, 'logs', train_name) \
            == read_file(appended, 'logs', val_name)


def test_append_writes_a_repeated_pair_once(tmp_path, zips, zips_b):
    """The same run twice adds the pairs once only."""
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    assert run_b(zips_b, out, extra=['--append']) == 0
    once = list_bytes(out)
    assert read_list(str(out), 'pairs.txt') == EXPECTED_MERGED
    assert run_b(zips_b, out, extra=['--append']) == 0
    assert read_list(str(out), 'pairs.txt') == EXPECTED_MERGED
    assert list_bytes(out) == once


def test_append_does_not_add_a_pair_again_in_the_other_order(tmp_path, zips):
    """A folder made with the old order gets no second copy of its pairs.

    The lists of an earlier release put the anchor image first in every pair.
    The same run with --append now writes some of those pairs the other way
    round. Such a pair is already in the lists, so the folder must not change.
    """
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    for path in extract_sample.list_paths(os.path.join(str(out), 'logs')):
        with open(path) as handle:
            lines = handle.read().splitlines()
        with open(path, 'w') as handle:
            handle.write(''.join(' '.join(line.split()[::-1]) + '\n' for line in lines))
    reversed_pairs = [' '.join(line.split()[::-1]) for line in EXPECTED_PAIRS]
    assert read_list(str(out), 'pairs.txt') == reversed_pairs
    before = snapshot(out)

    assert run(zips, out, extra=['--append']) == 0
    assert read_list(str(out), 'pairs.txt') == reversed_pairs
    assert snapshot(out) == before


def test_append_on_an_empty_folder_is_a_plain_run(tmp_path, zips):
    """--append needs no earlier run. The manifest then holds one run."""
    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--append']) == 0
    assert read_list(str(out), 'pairs.txt') == EXPECTED_PAIRS
    with open(os.path.join(str(out), 'sample_manifest.json')) as handle:
        manifest = json.load(handle)
    assert len(manifest['runs']) == 1
    assert manifest['runs'][0]['logs'] == LOGS


def test_a_second_run_without_append_refuses_and_changes_nothing(
        tmp_path, zips, zips_b):
    """Without --append the tool must not overwrite the lists of a run."""
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    before = list_bytes(out)
    before_npy = {path: read_file(path) for path in npy_files(out)}
    before_manifest = read_file(str(out), 'sample_manifest.json')

    with pytest.raises(ValueError) as error:
        run_b(zips_b, out)
    assert 'pairs.txt' in str(error.value)
    assert '--append' in str(error.value)

    assert list_bytes(out) == before
    assert {path: read_file(path) for path in npy_files(out)} == before_npy
    assert read_file(str(out), 'sample_manifest.json') == before_manifest


def test_append_merges_the_manifest(appended):
    """The manifest describes both runs and every file in the folder."""
    with open(os.path.join(appended, 'sample_manifest.json')) as handle:
        manifest = json.load(handle)

    runs = manifest['runs']
    assert len(runs) == 2
    assert runs[0]['logs'] == LOGS
    assert runs[0]['anchor'] == ANCHOR
    assert runs[0]['frames'] == FRAMES
    assert runs[0]['max_gap'] == 5
    assert runs[0]['source_zips'] == [log + '.zip' for log in LOGS]
    assert runs[1]['logs'] == LOGS_B
    assert runs[1]['anchor'] == ANCHOR_B
    assert runs[1]['frames'] == FRAMES_B
    assert runs[1]['max_gap'] == MAX_GAP_B
    assert runs[1]['source_zips'] == [log + '.zip' for log in LOGS_B]
    for run_record in runs:
        assert run_record['sonar'] == 0
        assert run_record['scene'] is None
        assert run_record['script_sha256']
        assert run_record['git_sha']

    # The keys outside runs describe the latest run and the whole folder.
    assert manifest['logs'] == LOGS_B
    assert manifest['anchor'] == ANCHOR_B
    assert manifest['max_gap'] == MAX_GAP_B
    assert [' '.join(pair) for pair in manifest['pairs']] == EXPECTED_MERGED

    on_disk = npy_files(appended)
    assert len(on_disk) == 45
    assert manifest['files'] == len(on_disk)
    assert len(manifest['md5']) == len(on_disk)
    assert sorted(os.path.join(appended, *path.split('/'))
                  for path in manifest['md5']) == on_disk
    total = 0
    for path, digest in manifest['md5'].items():
        data = read_file(appended, *path.split('/'))
        assert hashlib.md5(data).hexdigest() == digest, path
        total += len(data)
    assert manifest['total_bytes'] == total > 0
    # The files of the first run are still there, and still the same.
    assert 'logs/T_BPL_1/Sonar_0/S_10.npy' in manifest['md5']


def test_the_appended_sample_loads_through_the_dataloader(appended):
    """The release dataloader reads the merged sample."""
    import config
    from dataloader.sonardata import SonarData

    args = config.get_args([])
    args.datadir = appended
    args.phase = 'test'
    dataset = SonarData(args)
    assert len(dataset) == 25
    names = (dataset.imf1s + dataset.imf2s + dataset.posf1s + dataset.posf2s
             + dataset.metaf1s + dataset.metaf2s)
    assert len(names) == 150
    for name in names:
        assert os.path.isfile(name), name


# --------------------------------------------------------------------------
# --append: the folder must match its manifest, and stay whole
# --------------------------------------------------------------------------

# The keys that a manifest of the first release holds for its one run.
LEGACY_RUN_KEYS = ('anchor', 'frames', 'git_sha', 'logs', 'max_gap', 'scene',
                   'script_sha256', 'sonar', 'source_zips')


def test_append_stops_before_writing_when_a_file_is_gone(tmp_path, zips, zips_b):
    """A folder that lost a file of its manifest is left as it is."""
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    gone = os.path.join(str(out), 'logs', 'T_BPH_1', 'Sonar_0', 'S_10.npy')
    assert os.path.isfile(gone)
    os.remove(gone)
    before = snapshot(out)

    with pytest.raises(ValueError) as error:
        run_b(zips_b, out, extra=['--append'])
    assert 'S_10.npy' in str(error.value)
    # No new file, no rewritten list, no touched byte.
    assert snapshot(out) == before


def test_append_needs_the_manifest_of_the_earlier_run(tmp_path, zips, zips_b):
    """Without the manifest the tool cannot count the files of the folder."""
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    os.remove(os.path.join(str(out), 'sample_manifest.json'))
    before = snapshot(out)

    with pytest.raises(ValueError) as error:
        run_b(zips_b, out, extra=['--append'])
    assert 'sample_manifest.json' in str(error.value)
    assert snapshot(out) == before


def test_a_repeated_append_writes_nothing(tmp_path, zips, zips_b, capsys):
    """The same append twice leaves every file as the first run left it."""
    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--preview']) == 0
    assert run_b(zips_b, out, extra=['--append', '--preview']) == 0
    before = snapshot(out)

    capsys.readouterr()
    assert run_b(zips_b, out, extra=['--append', '--preview']) == 0
    assert '0 data file(s) written' in capsys.readouterr().out
    assert snapshot(out) == before


def test_append_reads_a_manifest_without_runs(tmp_path, zips, zips_b):
    """A manifest of the first release holds no runs key.

    Its own keys describe that one run, so they become the first record.
    """
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    path = os.path.join(str(out), 'sample_manifest.json')
    with open(path) as handle:
        legacy = json.load(handle)
    del legacy['runs']
    with open(path, 'w') as handle:
        json.dump(legacy, handle, indent=2, sort_keys=True)
        handle.write('\n')

    assert run_b(zips_b, out, extra=['--append']) == 0
    with open(path) as handle:
        manifest = json.load(handle)
    assert len(manifest['runs']) == 2
    assert manifest['runs'][0] == {key: legacy[key] for key in LEGACY_RUN_KEYS}
    assert manifest['runs'][0]['logs'] == LOGS
    assert manifest['runs'][1]['logs'] == LOGS_B
    assert manifest['runs'][1]['frames'] == FRAMES_B
    assert read_list(str(out), 'pairs.txt') == EXPECTED_MERGED


# --------------------------------------------------------------------------
# the manifest counts the data files
# --------------------------------------------------------------------------

def test_the_manifest_counts_the_data_files_only(appended):
    """md5, files and total_bytes cover the .npy data files, nothing else."""
    with open(os.path.join(appended, 'sample_manifest.json')) as handle:
        manifest = json.load(handle)
    data = npy_files(appended)
    assert manifest['files'] == len(data) == 45
    assert len(manifest['md5']) == len(data)
    for path in manifest['md5']:
        assert path.startswith('logs/'), path
        assert path.endswith('.npy'), path
    assert manifest['total_bytes'] == sum(os.path.getsize(p) for p in data)


def test_the_previews_are_listed_apart_from_the_data_files(tmp_path, zips, zips_b):
    """A preview belongs under preview, never under md5."""
    out = tmp_path / 'sample'
    assert run(zips, out, extra=['--preview']) == 0
    assert run_b(zips_b, out, extra=['--append', '--preview']) == 0

    with open(os.path.join(str(out), 'sample_manifest.json')) as handle:
        manifest = json.load(handle)
    on_disk = sorted(os.listdir(os.path.join(str(out), 'preview')))
    assert len(on_disk) == 15
    assert sorted(manifest['preview']) == on_disk
    assert not [name for name in manifest['md5'] if not name.endswith('.npy')]
    assert manifest['files'] == len(npy_files(out)) == 45


# A third run of the second scene. Its frames overlap the frames of the second
# run, so five of its nine pairs are already in the lists and four are new.
FRAMES_C = [8, 12, 16]
EXPECTED_APPEND_C = [
    'logs/U_BPH_1/Sonar_0/S_16.npy logs/U_BPL_1/Sonar_0/S_12.npy',
    'logs/U_BPH_1/Sonar_0/S_12.npy logs/U_BPL_1/Sonar_0/S_16.npy',
    'logs/U_BPH_1/Sonar_0/S_16.npy logs/U_BPL_1/Sonar_0/S_16.npy',
    'logs/U_BPL_1/Sonar_0/S_12.npy logs/U_BPL_1/Sonar_0/S_16.npy',
]


def test_append_reads_only_the_files_of_the_new_pairs(tmp_path, zips, zips_b,
                                                      capsys):
    """A pair that is already in the lists costs no read and no write.

    The third run repeats five pairs and adds four. Only the four new lines
    name the files that the tool reads, so frame 4, which only the repeated
    pairs name, is not written again.
    """
    out = tmp_path / 'sample'
    assert run(zips, out) == 0
    assert run_b(zips_b, out, extra=['--append']) == 0
    before = snapshot(out)
    # The six files of frame 4: three kinds in each of the two logs.
    frame_four = sorted(rel for rel in before if rel.endswith('_4.npy'))
    assert len(frame_four) == 6

    capsys.readouterr()
    assert run_b(zips_b, out, extra=['--append'], frames=FRAMES_C) == 0
    printed = capsys.readouterr().out
    # Four new lines name four images: 12 data files, not the 18 of the plan.
    assert '12 data file(s) written' in printed

    assert read_list(str(out), 'pairs.txt') == EXPECTED_MERGED + EXPECTED_APPEND_C
    after = snapshot(out)
    for rel in frame_four:
        assert after[rel] == before[rel], rel
    assert len(npy_files(out)) == 51
