"""Tests for make_pairs.py, the C-SONIC pair-list generator.

The tests build a small synthetic dataset in a temporary directory. They read
no data from the real dataset and they load no torch module.
"""
import collections
import hashlib
import json
import os
import zipfile

import numpy as np
import pytest

import make_pairs

# Two synthetic scenes. Scene A holds one high-frequency log, one low-frequency
# log and one long-range log, so all four pair categories occur inside it.
SCENES = {'A': ['A_BPH_1', 'A_BPL_1', 'A_BPL_1_LR'],
          'B': ['B_BPL_1', 'B_BPL_1_LR']}
SHAPE = {'A': (3, 30), 'B': (2, 20)}
# A log that no scene lists. The generator ignores it. The inventory holds it,
# so --verify must check the lines that point to it.
EXTRA_LOG = ('C_BPL_1', 1, 5)

SCENE_ARGS = ['--scene', 'A=A_BPH_1,A_BPL_1,A_BPL_1_LR',
              '--scene', 'B=B_BPL_1,B_BPL_1_LR']


def log_files(log, n_sonars, n_frames):
    """Yield the relative names of the image, pose and metadata files."""
    for sonar in range(n_sonars):
        for frame in range(n_frames):
            yield '{}/Sonar_{}/S_{}.npy'.format(log, sonar, frame)
            yield '{}/Pose_{}/P_{}.npy'.format(log, sonar, frame)
            yield '{}/Metadata_{}/M_{}.npy'.format(log, sonar, frame)


def make_tree(root, shape=None, extra=EXTRA_LOG):
    """Write empty .npy files under root/logs. Returns the root as a string."""
    shape = SHAPE if shape is None else shape
    plan = [(log, shape[scene][0], shape[scene][1])
            for scene, logs in SCENES.items() for log in logs]
    if extra:
        plan.append(extra)
    for log, n_sonars, n_frames in plan:
        for sonar in range(n_sonars):
            for kind in ('Sonar', 'Pose', 'Metadata'):
                os.makedirs(os.path.join(str(root), 'logs', log,
                                         '{}_{}'.format(kind, sonar)),
                            exist_ok=True)
        for name in log_files(log, n_sonars, n_frames):
            open(os.path.join(str(root), 'logs', name), 'wb').close()
    return str(root)


def make_zip(path, log, n_sonars, n_frames, top=None):
    """Write a zip in the dataset layout. Returns the path as a string."""
    top = log if top is None else top
    with zipfile.ZipFile(str(path), 'w') as handle:
        handle.writestr(top + '/', b'')
        for name in log_files(log, n_sonars, n_frames):
            handle.writestr(name.replace(log + '/', top + '/', 1), b'')
    return str(path)


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    """A synthetic dataset root, built once for the read-only tests."""
    return make_tree(tmp_path_factory.mktemp('data'))


@pytest.fixture(scope='module')
def inventory(tree):
    return make_pairs.Inventory.from_dir(tree)


def sonic_dir(root, lines, val_lines, pose_lines=None, val_pose_lines=None):
    """Write a folder that imitates the SONIC pair lists. Returns the path.

    The SONIC release ships pairs.txt, pairs_pos.txt and pairs_val.txt. It
    ships no pairs_pos_val.txt. Give pose_lines=False to leave the pose list
    out, or val_pose_lines to add the validation pose list.
    """
    os.makedirs(str(root), exist_ok=True)
    with open(os.path.join(str(root), 'pairs.txt'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    with open(os.path.join(str(root), 'pairs_val.txt'), 'w') as handle:
        handle.write('\n'.join(val_lines) + '\n')
    if pose_lines is None:
        pose_lines = make_pairs.derive_lines(lines, 'pose')
    if pose_lines is not False:
        with open(os.path.join(str(root), 'pairs_pos.txt'), 'w') as handle:
            # The SONIC file has no final newline.
            handle.write('\n'.join(pose_lines))
    if val_pose_lines:
        with open(os.path.join(str(root), 'pairs_pos_val.txt'), 'w') as handle:
            handle.write('\n'.join(val_pose_lines) + '\n')
    return str(root)


def read_lines(path):
    with open(path) as handle:
        return handle.read().splitlines()


# --------------------------------------------------------------------------
# path convention
# --------------------------------------------------------------------------

@pytest.mark.parametrize('log', ['ELC1_BPH_1', 'S_5', 'S_Sonar_1_BPL_LR'])
def test_image_path_round_trip(log):
    """parse_image_path inverts image_path, also for a log named like a file."""
    path = make_pairs.image_path(log, 3, 857)
    assert path == 'logs/{}/Sonar_3/S_857.npy'.format(log)
    assert make_pairs.parse_image_path(path) == (log, 3, 857)


@pytest.mark.parametrize('log', ['ELC1_BPH_1', 'S_5', 'S_Sonar_1_BPL_LR'])
def test_pose_and_meta_paths(log):
    """The pose and metadata paths change the folder and the file prefix only."""
    path = make_pairs.image_path(log, 3, 857)
    assert make_pairs.pose_path_from_image(path) == \
        'logs/{}/Pose_3/P_857.npy'.format(log)
    assert make_pairs.meta_path_from_image(path) == \
        'logs/{}/Metadata_3/M_857.npy'.format(log)


@pytest.mark.parametrize('bad', [
    'logs/L/Sonar_3/S_857.txt',
    'logs/L/Sonar_3/P_857.npy',
    'logs/L/Pose_3/S_857.npy',
    'logs/L/Sonar_x/S_857.npy',
    'logs/L/Sonar_3/S_x.npy',
    'L/Sonar_3/S_857.npy',
    'data/L/Sonar_3/S_857.npy',
    'logs/L/Sonar_3/sub/S_857.npy',
    '',
])
def test_parse_image_path_rejects_malformed(bad):
    """A path that does not follow the convention raises ValueError."""
    with pytest.raises(ValueError):
        make_pairs.parse_image_path(bad)


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------

def test_from_dir_matches_from_zips(tmp_path, tree, inventory):
    """A zip and a directory of the same data give the same inventory."""
    zips = []
    for scene, logs in SCENES.items():
        for log in logs:
            zips.append(make_zip(tmp_path / (log + '.zip'), log, *SHAPE[scene]))
    zips.append(make_zip(tmp_path / (EXTRA_LOG[0] + '.zip'), *EXTRA_LOG))
    from_zips = make_pairs.Inventory.from_zips(zips)

    assert from_zips == inventory
    assert from_zips.logs == sorted(
        [log for logs in SCENES.values() for log in logs] + [EXTRA_LOG[0]])
    assert inventory.has('logs/A_BPH_1/Sonar_2/S_29.npy')
    assert inventory.has('logs/A_BPH_1/Pose_2/P_29.npy')
    assert inventory.has('logs/A_BPH_1/Metadata_2/M_29.npy')
    assert not inventory.has('logs/A_BPH_1/Sonar_2/S_30.npy')
    assert not inventory.has('logs/A_BPH_1/Sonar_3/S_0.npy')
    assert not inventory.has('logs/NO_SUCH_BPL/Sonar_0/S_0.npy')


def test_merge_joins_two_inventories(tmp_path, tree, inventory):
    """Two part inventories merge into the inventory of the whole tree."""
    first = make_pairs.Inventory.from_zips(
        [make_zip(tmp_path / (log + '.zip'), log, *SHAPE['A'])
         for log in SCENES['A']])
    rest = [make_zip(tmp_path / (log + '.zip'), log, *SHAPE['B'])
            for log in SCENES['B']]
    rest.append(make_zip(tmp_path / (EXTRA_LOG[0] + '.zip'), *EXTRA_LOG))
    second = make_pairs.Inventory.from_zips(rest)

    assert first != inventory
    assert make_pairs.Inventory.merge(first, second) == inventory


def test_from_dir_missing_pose_file_raises(tmp_path):
    """A pose file that the sonar folder does not match is an error."""
    root = make_tree(tmp_path / 'data')
    os.remove(os.path.join(root, 'logs', 'A_BPL_1', 'Pose_1', 'P_7.npy'))
    with pytest.raises(ValueError) as excinfo:
        make_pairs.Inventory.from_dir(root)
    assert 'A_BPL_1' in str(excinfo.value)
    assert 'Pose_1' in str(excinfo.value)


def test_from_zips_missing_meta_file_raises(tmp_path):
    """The same check runs on a zip."""
    path = str(tmp_path / 'A_BPH_1.zip')
    with zipfile.ZipFile(path, 'w') as handle:
        for name in log_files('A_BPH_1', 1, 4):
            if name.endswith('Metadata_0/M_2.npy'):
                continue
            handle.writestr(name, b'')
    with pytest.raises(ValueError) as excinfo:
        make_pairs.Inventory.from_zips([path])
    assert 'Metadata_0' in str(excinfo.value)


def test_from_zips_rejects_wrong_top_level(tmp_path):
    """The folder inside the zip must carry the name of the zip."""
    path = make_zip(tmp_path / 'A_BPH_1.zip', 'A_BPH_1', 1, 3, top='OTHER_LOG')
    with pytest.raises(ValueError) as excinfo:
        make_pairs.Inventory.from_zips([path])
    assert 'OTHER_LOG' in str(excinfo.value)


# --------------------------------------------------------------------------
# scene table
# --------------------------------------------------------------------------

def test_check_scenes_returns_the_common_sets(inventory):
    """check_scenes gives the sonar and frame set that the scene shares."""
    table = make_pairs.check_scenes(inventory, SCENES)
    assert list(table['A'][0]) == [0, 1, 2]
    assert list(table['A'][1]) == list(range(30))
    assert list(table['B'][0]) == [0, 1]
    assert list(table['B'][1]) == list(range(20))


def test_check_scenes_rejects_missing_log(inventory):
    with pytest.raises(ValueError) as excinfo:
        make_pairs.check_scenes(inventory, {'A': ['A_BPH_1', 'A_NOT_THERE']})
    assert 'A_NOT_THERE' in str(excinfo.value)


def test_check_scenes_rejects_frame_mismatch(tmp_path):
    """Logs of one scene must hold the same frames."""
    root = make_tree(tmp_path / 'data', shape={'A': (3, 30), 'B': (2, 20)})
    for kind, prefix in (('Sonar', 'S'), ('Pose', 'P'), ('Metadata', 'M')):
        os.remove(os.path.join(root, 'logs', 'A_BPL_1',
                               '{}_2'.format(kind), '{}_29.npy'.format(prefix)))
    inv = make_pairs.Inventory.from_dir(root)
    with pytest.raises(ValueError) as excinfo:
        make_pairs.check_scenes(inv, SCENES)
    assert 'A' in str(excinfo.value) and 'A_BPL_1' in str(excinfo.value)


def test_parse_scene_args():
    """--scene NAME=LOG,LOG replaces the default table."""
    assert make_pairs.parse_scene_args(['A=A_BPH_1,A_BPL_1', 'B=B_BPL_1']) == \
        {'A': ['A_BPH_1', 'A_BPL_1'], 'B': ['B_BPL_1']}
    assert make_pairs.parse_scene_args([]) == make_pairs.DEFAULT_SCENES
    with pytest.raises(ValueError):
        make_pairs.parse_scene_args(['no-equals-sign'])


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def test_allocate_is_proportional_and_exact():
    """Largest-remainder rounding keeps the total exact."""
    assert make_pairs.allocate(10, [1, 1, 1]) == [4, 3, 3]
    assert make_pairs.allocate(100, [1, 3]) == [25, 75]
    weights = [33780, 33780, 52830]
    parts = make_pairs.allocate(255122, weights)
    assert sum(parts) == 255122
    for part, weight in zip(parts, weights):
        assert abs(part - 255122 * weight / sum(weights)) < 1.0
    assert make_pairs.allocate(0, [1, 2]) == [0, 0]


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def pairs(inventory):
    return make_pairs.generate(inventory, SCENES, 4000, seed=7)


def logs_of(pair):
    return (make_pairs.parse_image_path(pair[0])[0],
            make_pairs.parse_image_path(pair[1])[0])


def test_generate_exact_count_and_no_duplicates(pairs):
    assert len(pairs) == 4000
    assert len(set(pairs)) == 4000


def test_generate_paths_exist(pairs, inventory):
    """Every image, pose and metadata path of a new pair is in the inventory."""
    for pair in pairs:
        for path in pair:
            assert inventory.has(path)
            assert inventory.has(make_pairs.pose_path_from_image(path))
            assert inventory.has(make_pairs.meta_path_from_image(path))


def test_generate_stays_inside_one_scene(pairs):
    """A pair never joins two scenes."""
    scene_of = {log: scene for scene, logs in SCENES.items() for log in logs}
    for pair in pairs:
        first, second = logs_of(pair)
        assert scene_of[first] == scene_of[second]


def test_generate_gaps(pairs):
    """The gap is in [1, gap_max] inside a log and in [0, gap_max] across logs.

    The two counters guard the test against passing on an empty selection.
    """
    same_log = cross_log = zero_gap = 0
    for pair in pairs:
        log1, sonar1, frame1 = make_pairs.parse_image_path(pair[0])
        log2, sonar2, frame2 = make_pairs.parse_image_path(pair[1])
        gap = frame2 - frame1
        assert gap <= 9
        if log1 == log2:
            same_log += 1
            assert gap >= 1
        else:
            cross_log += 1
            assert gap >= 0
            zero_gap += gap == 0
    assert same_log > 500 and cross_log > 500 and zero_gap > 100


def test_generate_sonars_are_independent(pairs):
    """The two sonars are drawn apart, so they agree only sometimes."""
    same = sum(make_pairs.parse_image_path(p[0])[1] ==
               make_pairs.parse_image_path(p[1])[1] for p in pairs)
    assert 0 < same < len(pairs)


def test_generate_all_categories_and_balanced_logs(pairs):
    """All four categories occur and every log combination gets its share."""
    counts = collections.Counter(make_pairs.categorise(p, SCENES) for p in pairs)
    assert set(counts) == {'same-log', 'cross-frequency', 'cross-range',
                           'cross-frequency-range'}
    combos = collections.Counter(logs_of(p) for p in pairs)
    for logs in SCENES.values():
        seen = [combos[(a, b)] for a in logs for b in logs]
        mean = sum(seen) / float(len(seen))
        assert mean > 0
        assert min(seen) > 0.6 * mean
        assert max(seen) < 1.4 * mean


def test_generate_is_deterministic(inventory, pairs):
    assert make_pairs.generate(inventory, SCENES, 4000, seed=7) == pairs
    assert make_pairs.generate(inventory, SCENES, 4000, seed=8) != pairs


# The first pairs of the draw for seed 7 on the synthetic dataset, written down
# once. They pin the order of the list: a sort, a mix or a different order of
# the draws inside generate() changes them. numpy promises that
# default_rng(seed) gives the same numbers again.
GOLDEN_FIRST = [('logs/A_BPL_1/Sonar_1/S_3.npy', 'logs/A_BPH_1/Sonar_2/S_6.npy'),
                ('logs/A_BPL_1_LR/Sonar_0/S_26.npy', 'logs/A_BPL_1/Sonar_0/S_28.npy'),
                ('logs/A_BPL_1_LR/Sonar_2/S_18.npy', 'logs/A_BPL_1_LR/Sonar_0/S_23.npy')]
GOLDEN_LAST = ('logs/B_BPL_1/Sonar_0/S_2.npy', 'logs/B_BPL_1/Sonar_0/S_6.npy')


def test_generate_keeps_the_order_of_the_draw(pairs):
    """The list holds the pairs in the order of the draw, in no other order."""
    assert pairs[:3] == GOLDEN_FIRST
    assert pairs[-1] == GOLDEN_LAST
    assert pairs != sorted(pairs)
    assert pairs != sorted(pairs, reverse=True)


def test_generate_allocates_over_scenes(inventory):
    """The scenes share the pairs in proportion to sonars times frames."""
    got = make_pairs.generate(inventory, SCENES, 1300, seed=1)
    per_scene = collections.Counter(
        'A' if logs_of(p)[0].startswith('A') else 'B' for p in got)
    assert per_scene['A'] + per_scene['B'] == 1300
    assert per_scene['A'] == 900 and per_scene['B'] == 400


def test_generate_respects_gap_max(inventory):
    got = make_pairs.generate(inventory, SCENES, 500, seed=3, gap_max=1)
    gaps = {make_pairs.parse_image_path(p[1])[2] -
            make_pairs.parse_image_path(p[0])[2] for p in got}
    assert gaps == {0, 1}


def test_categorise():
    def pair(log1, log2):
        return (make_pairs.image_path(log1, 0, 1), make_pairs.image_path(log2, 0, 2))
    assert make_pairs.categorise(pair('A_BPL_1', 'A_BPL_1')) == 'same-log'
    assert make_pairs.categorise(pair('A_BPH_1', 'A_BPL_1')) == 'cross-frequency'
    assert make_pairs.categorise(pair('A_BPL_1', 'A_BPL_1_LR')) == 'cross-range'
    assert make_pairs.categorise(pair('A_BPH_1', 'A_BPL_1_LR')) == \
        'cross-frequency-range'
    assert make_pairs.categorise(pair('A_BPH_1', 'A_BPL_1'), SCENES) == \
        'cross-frequency'
    with pytest.raises(ValueError):
        make_pairs.categorise(pair('A_BPH_1', 'B_BPL_1'), SCENES)


def test_pair_stats(pairs):
    """The statistics count every pair once."""
    stats = make_pairs.pair_stats(pairs, SCENES)
    assert sum(stats['categories'].values()) == len(pairs)
    assert sum(stats['gap_histogram'].values()) == len(pairs)
    assert stats['gap_histogram'].get('0', 0) > 0
    assert 0 < stats['zero_motion'] < len(pairs)


# --------------------------------------------------------------------------
# derived and merged lists
# --------------------------------------------------------------------------

def test_derived_lines_are_parallel():
    lines = ['logs/L/Sonar_1/S_2.npy logs/L/Sonar_3/S_4.npy']
    assert make_pairs.derive_lines(lines, 'pose') == \
        ['logs/L/Pose_1/P_2.npy logs/L/Pose_3/P_4.npy']
    assert make_pairs.derive_lines(lines, 'meta') == \
        ['logs/L/Metadata_1/M_2.npy logs/L/Metadata_3/M_4.npy']


def test_read_sonic_list_rejects_a_malformed_line(tmp_path):
    path = str(tmp_path / 'pairs.txt')
    with open(path, 'w') as handle:
        handle.write('logs/L/Sonar_1/S_2.npy logs/L/Sonar_3/S_4.npy\nonly_one\n')
    with pytest.raises(ValueError) as excinfo:
        make_pairs.read_sonic_list(path)
    assert ':2' in str(excinfo.value)


def test_check_sonic_pose_list_detects_a_wrong_line(tmp_path):
    image_lines = ['logs/L/Sonar_{i}/S_{i}.npy logs/L/Sonar_{i}/S_9.npy'.format(i=i)
                   for i in range(3)]
    pose_lines = make_pairs.derive_lines(image_lines, 'pose')
    make_pairs.check_sonic_pose_list(image_lines, list(pose_lines), 'x')
    wrong = list(pose_lines)
    wrong[1] = 'logs/L/Pose_1/P_1.npy logs/L/Pose_1/P_8.npy'
    with pytest.raises(ValueError) as excinfo:
        make_pairs.check_sonic_pose_list(image_lines, wrong, 'x')
    assert 'line 2' in str(excinfo.value)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

SONIC_TRAIN = ['logs/FOREIGN_BPL/Sonar_0/S_{}.npy logs/FOREIGN_BPL/Sonar_1/S_{}.npy'
               .format(i, i + 2) for i in range(5)]
SONIC_TRAIN += ['logs/C_BPL_1/Sonar_0/S_1.npy logs/C_BPL_1/Sonar_0/S_3.npy']
SONIC_VAL = ['logs/FOREIGN_BPL/Sonar_2/S_{}.npy logs/FOREIGN_BPL/Sonar_2/S_{}.npy'
             .format(i, i + 1) for i in range(3)]


@pytest.fixture(scope='module')
def run_out(tmp_path_factory, tree):
    """Run the tool once with --verify. Returns (code, out_dir)."""
    out = str(tmp_path_factory.mktemp('out'))
    sonic = sonic_dir(tmp_path_factory.mktemp('sonic'), SONIC_TRAIN, SONIC_VAL,
                      pose_lines=make_pairs.derive_lines(SONIC_TRAIN, 'pose'))
    code = make_pairs.main(['--root', tree, '--out', out, '--sonic-dir', sonic,
                            '--n-train', '400', '--n-val', '80', '--seed', '5',
                            '--verify'] + SCENE_ARGS)
    return code, out


def test_cli_writes_twelve_lists(run_out):
    """The tool writes the six new lists, the six merged lists and a manifest."""
    code, out = run_out
    assert code == 0
    assert sorted(os.listdir(out)) == sorted(
        list(make_pairs.ALL_NAMES) + [make_pairs.MANIFEST_NAME])
    for name in make_pairs.ALL_NAMES:
        with open(os.path.join(out, name), 'rb') as handle:
            assert handle.read().endswith(b'\n')


def test_cli_line_counts(run_out):
    """The three lists of a split agree in length, cross plus SONIC."""
    _, out = run_out
    for split, names in make_pairs.CROSS_NAMES.items():
        counts = {len(read_lines(os.path.join(out, name))) for name in names}
        assert counts == {400 if split == 'train' else 80}
    expected = {'train': 400 + len(SONIC_TRAIN), 'val': 80 + len(SONIC_VAL)}
    for split, names in make_pairs.MERGED_NAMES.items():
        counts = {len(read_lines(os.path.join(out, name))) for name in names}
        assert counts == {expected[split]}


def test_cli_merged_keeps_sonic_lines_first_then_the_new_ones(run_out):
    """The SONIC lines come first, unchanged and in their original order."""
    _, out = run_out
    merged = read_lines(os.path.join(out, 'pairs.txt'))
    cross = read_lines(os.path.join(out, 'pairs_cross.txt'))
    assert merged[:len(SONIC_TRAIN)] == SONIC_TRAIN
    assert merged[len(SONIC_TRAIN):] == cross
    assert read_lines(os.path.join(out, 'pairs_val.txt'))[:len(SONIC_VAL)] == SONIC_VAL
    # The pose list of the val split is derived, because SONIC has none.
    merged_pos = read_lines(os.path.join(out, 'pairs_pos_val.txt'))
    assert merged_pos[:len(SONIC_VAL)] == make_pairs.derive_lines(SONIC_VAL, 'pose')


def test_cli_merged_tail_equals_the_cross_file_byte_for_byte(run_out):
    _, out = run_out
    for split in ('train', 'val'):
        for merged_name, cross_name in zip(make_pairs.MERGED_NAMES[split],
                                           make_pairs.CROSS_NAMES[split]):
            with open(os.path.join(out, cross_name), 'rb') as handle:
                cross = handle.read()
            with open(os.path.join(out, merged_name), 'rb') as handle:
                assert handle.read().endswith(cross)


def test_cli_lists_are_parallel(run_out):
    """Line n of the pose and metadata lists describes line n of the images."""
    _, out = run_out
    for split in ('train', 'val'):
        images, poses, metas = (read_lines(os.path.join(out, name))
                                for name in make_pairs.MERGED_NAMES[split])
        assert poses == make_pairs.derive_lines(images, 'pose')
        assert metas == make_pairs.derive_lines(images, 'meta')


def test_cli_does_not_shuffle(run_out, tree):
    """The new lines keep the order the generator produced."""
    _, out = run_out
    cross = read_lines(os.path.join(out, 'pairs_cross.txt'))
    assert cross == ['{} {}'.format(*p) for p in make_pairs.generate(
        make_pairs.Inventory.from_dir(tree), SCENES, 400, seed=5)]


def read_manifest(out):
    with open(os.path.join(out, make_pairs.MANIFEST_NAME)) as handle:
        return json.load(handle)


def test_cli_manifest(run_out):
    """The manifest counts match the files, and the md5 sums match too."""
    _, out = run_out
    manifest = read_manifest(out)
    assert manifest['seed'] == 5
    assert manifest['scenes'] == SCENES
    for split, expected_new in (('train', 400), ('val', 80)):
        counts = manifest['counts'][split]
        assert counts['new'] == expected_new
        assert counts['sonic'] + counts['new'] == counts['total']
        name = make_pairs.MERGED_NAMES[split][0]
        assert counts['total'] == len(read_lines(os.path.join(out, name)))
    assert sorted(manifest['md5']) == sorted(make_pairs.ALL_NAMES)
    for name, digest in manifest['md5'].items():
        with open(os.path.join(out, name), 'rb') as handle:
            assert hashlib.md5(handle.read()).hexdigest() == digest
    stats = manifest['new_pairs']['train']
    assert sum(stats['categories'].values()) == 400
    assert sum(int(k) * 0 + v for k, v in stats['gap_histogram'].items()) == 400
    assert sum(manifest['per_log_lines']['train'].values()) == \
        400 + len(SONIC_TRAIN)


def test_cli_verify_report(run_out):
    """The report names the log that the inventory does not hold."""
    _, out = run_out
    report = read_manifest(out)['verify']
    assert report['ok'] is True
    assert report['missing_count'] == 0
    assert list(report['not_verifiable']) == ['FOREIGN_BPL']
    assert report['not_verifiable']['FOREIGN_BPL'] == \
        len(SONIC_TRAIN) - 1 + len(SONIC_VAL)
    assert 'A_BPH_1' in report['verified_logs']
    assert report['tail_ok'] is True
    assert report['reasons'] == []


def test_cli_verify_finds_a_deleted_file(tmp_path):
    """Removing a file that a list names makes --verify fail with code 2.

    The file belongs to C_BPL_1, which no scene lists, so the deletion reaches
    the verification and not the consistency check of the inventory.
    """
    root = make_tree(tmp_path / 'data')
    out = str(tmp_path / 'out')
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL)
    argv = ['--root', root, '--out', out, '--sonic-dir', sonic,
            '--n-train', '100', '--n-val', '20', '--verify'] + SCENE_ARGS
    assert make_pairs.main(argv) == 0

    for kind, prefix in (('Sonar', 'S'), ('Pose', 'P'), ('Metadata', 'M')):
        os.remove(os.path.join(root, 'logs', 'C_BPL_1', '{}_0'.format(kind),
                               '{}_1.npy'.format(prefix)))
    assert make_pairs.main(argv) == 2


def test_cli_verify_finds_a_deleted_pose_file(tmp_path):
    """One deleted pose file stops the run in the consistency check."""
    root = make_tree(tmp_path / 'data')
    os.remove(os.path.join(root, 'logs', 'B_BPL_1', 'Pose_0', 'P_3.npy'))
    with pytest.raises(ValueError):
        make_pairs.main(['--root', root, '--out', str(tmp_path / 'out'),
                         '--n-train', '10', '--n-val', '2'] + SCENE_ARGS)


def test_cli_dry_run_writes_nothing(tmp_path, tree, capsys):
    out = tmp_path / 'out'
    code = make_pairs.main(['--root', tree, '--out', str(out), '--n-train', '50',
                            '--n-val', '10', '--dry-run'] + SCENE_ARGS)
    assert code == 0
    assert not out.exists() or os.listdir(str(out)) == []
    assert '50' in capsys.readouterr().out


def test_cli_without_sonic_dir_merged_equals_cross(tmp_path, tree):
    out = str(tmp_path / 'out')
    assert make_pairs.main(['--root', tree, '--out', out, '--n-train', '60',
                            '--n-val', '12'] + SCENE_ARGS) == 0
    for split in ('train', 'val'):
        for merged_name, cross_name in zip(make_pairs.MERGED_NAMES[split],
                                           make_pairs.CROSS_NAMES[split]):
            assert read_lines(os.path.join(out, merged_name)) == \
                read_lines(os.path.join(out, cross_name))


def test_cli_refuses_to_write_into_the_sonic_dir(tmp_path, tree):
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL)
    with pytest.raises(ValueError):
        make_pairs.main(['--root', tree, '--out', sonic, '--sonic-dir', sonic,
                         '--n-train', '10', '--n-val', '2'] + SCENE_ARGS)


def test_cli_needs_a_source():
    with pytest.raises(SystemExit):
        make_pairs.main(['--out', '/tmp/does-not-matter'])


def test_source_never_shuffles():
    """The dataloader shuffles. make_pairs must not, so the word is absent."""
    with open(make_pairs.__file__.replace('.pyc', '.py')) as handle:
        assert 'shuffle' not in handle.read()


# --------------------------------------------------------------------------
# round 2: capacity and input checks
# --------------------------------------------------------------------------

TINY_SCENES = {'T': ['T_BPH_1', 'T_BPL_1']}


@pytest.fixture(scope='module')
def tiny_inventory(tmp_path_factory):
    """Two logs, one sonar, three frames. You can count the capacity by hand."""
    root = str(tmp_path_factory.mktemp('tiny'))
    for log in TINY_SCENES['T']:
        for name in log_files(log, 1, 3):
            path = os.path.join(root, 'logs', name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, 'wb').close()
    return make_pairs.Inventory.from_dir(root)


def tiny_pairs(gap_max=1):
    """Every pair that the tiny scene can give, written out by hand."""
    expected = set()
    for first in TINY_SCENES['T']:
        for second in TINY_SCENES['T']:
            gaps = range(1, gap_max + 1) if first == second else range(0, gap_max + 1)
            for frame in range(3):
                for gap in gaps:
                    if frame + gap < 3:
                        expected.add((make_pairs.image_path(first, 0, frame),
                                      make_pairs.image_path(second, 0, frame + gap)))
    return expected


def test_scene_capacity(tiny_inventory):
    """The capacity counts the unique pairs of a scene.

    The tiny scene holds two logs, one sonar and the frames 0, 1 and 2. Inside
    one log the gap is 1, so frame1 is 0 or 1: 2 pairs per log, 4 for the two
    logs. Across the two logs the gap is 0 or 1: 3 + 2 = 5 pairs for each of
    the 2 ordered log pairs, so 10. The total is 14.
    """
    frames = tiny_inventory.frames('T_BPH_1', 0)
    assert len(tiny_pairs()) == 14
    assert make_pairs.scene_capacity(2, 1, frames, 1) == 14
    # Every sonar of the first image meets every sonar of the second image.
    assert make_pairs.scene_capacity(2, 2, frames, 1) == 56
    assert make_pairs.scene_capacity(1, 1, frames, 1) == 2


def test_generate_rejects_a_request_above_the_capacity(tiny_inventory):
    with pytest.raises(ValueError) as excinfo:
        make_pairs.generate(tiny_inventory, TINY_SCENES, 15, seed=1, gap_max=1)
    assert '15' in str(excinfo.value) and '14' in str(excinfo.value)
    assert 'T' in str(excinfo.value)


def test_generate_at_the_capacity_returns_every_pair_once(tiny_inventory):
    got = make_pairs.generate(tiny_inventory, TINY_SCENES, 14, seed=1, gap_max=1)
    assert len(got) == 14
    assert set(got) == tiny_pairs()


def test_generate_scene_stops_when_no_draw_is_new(tiny_inventory):
    """The guard stops a draw that can never reach the count."""
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        make_pairs._generate_scene(TINY_SCENES['T'], np.array([0]),
                                   tiny_inventory.frames('T_BPH_1', 0), 20, rng, 1)


def test_generate_rejects_a_gap_max_below_one(inventory):
    """A gap of zero inside one log would pair an image with itself."""
    with pytest.raises(ValueError):
        make_pairs.generate(inventory, SCENES, 10, seed=1, gap_max=0)
    with pytest.raises(ValueError):
        make_pairs.generate(inventory, SCENES, 10, seed=1, gap_max=-1)


def test_generate_rejects_a_negative_count(inventory):
    with pytest.raises(ValueError):
        make_pairs.generate(inventory, SCENES, -1, seed=1)


def test_check_scenes_rejects_a_duplicate_log(inventory):
    with pytest.raises(ValueError) as excinfo:
        make_pairs.check_scenes(inventory, {'A': ['A_BPH_1', 'A_BPH_1']})
    assert 'A_BPH_1' in str(excinfo.value)


def test_check_scenes_rejects_a_log_in_two_scenes(inventory):
    with pytest.raises(ValueError) as excinfo:
        make_pairs.check_scenes(inventory, {'A': ['A_BPH_1', 'A_BPL_1'],
                                            'B': ['A_BPH_1', 'B_BPL_1']})
    assert 'A_BPH_1' in str(excinfo.value) and 'scenes' in str(excinfo.value)


# [2, -1] is the case that needs the check: the weights add up to a positive
# number, so without the check the share of the second scene comes out negative
# and nothing stops it.
@pytest.mark.parametrize('weights', [[1, -1], [2, -1], [1, float('nan')],
                                     [1, float('inf')], [0, 0]])
def test_allocate_rejects_bad_weights(weights):
    """A weight that is not a number of zero or more is an error."""
    with pytest.raises(ValueError) as excinfo:
        make_pairs.allocate(10, weights)
    assert 'weight' in str(excinfo.value)


def test_allocate_rejects_a_negative_total():
    with pytest.raises(ValueError):
        make_pairs.allocate(-1, [1, 1])


# --------------------------------------------------------------------------
# round 2: the SONIC pose lists stay as they are
# --------------------------------------------------------------------------

def run_with_sonic(tmp_path, tree, sonic, extra=()):
    """Run the tool with a SONIC folder. Returns the output folder."""
    out = str(tmp_path / 'out')
    code = make_pairs.main(['--root', tree, '--out', out, '--sonic-dir', sonic,
                            '--n-train', '120', '--n-val', '30', '--seed', '5',
                            '--verify'] + SCENE_ARGS + list(extra))
    assert code == 0
    return out


def test_sonic_pose_lines_are_kept_byte_for_byte(tmp_path, tree):
    """A SONIC pose line with two spaces passes the check and stays as it is."""
    pose_lines = make_pairs.derive_lines(SONIC_TRAIN, 'pose')
    pose_lines[1] = pose_lines[1].replace(' ', '  ')
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL,
                      pose_lines=pose_lines)
    out = run_with_sonic(tmp_path, tree, sonic)

    merged = read_lines(os.path.join(out, 'pairs_pos.txt'))
    assert merged[:len(pose_lines)] == pose_lines
    assert '  ' in merged[1]
    # The metadata list is always derived, so it holds one space.
    assert '  ' not in read_lines(os.path.join(out, 'pairs_meta.txt'))[1]
    assert read_manifest(out)['sonic_pose_lists'] == {'train': 'kept',
                                                      'val': 'derived'}


def test_a_missing_sonic_pose_list_raises(tmp_path, tree):
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL, pose_lines=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        make_pairs.main(['--root', tree, '--out', str(tmp_path / 'out'),
                         '--sonic-dir', sonic, '--n-train', '10', '--n-val', '2']
                        + SCENE_ARGS)
    assert os.path.join(sonic, 'pairs_pos.txt') in str(excinfo.value)


def test_a_sonic_val_pose_list_is_kept_when_it_exists(tmp_path, tree):
    val_pose = make_pairs.derive_lines(SONIC_VAL, 'pose')
    val_pose[0] = val_pose[0].replace(' ', '   ')
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL,
                      val_pose_lines=val_pose)
    out = run_with_sonic(tmp_path, tree, sonic)

    assert read_lines(os.path.join(out, 'pairs_pos_val.txt'))[:len(val_pose)] == \
        val_pose
    assert read_manifest(out)['sonic_pose_lists'] == {'train': 'kept',
                                                      'val': 'kept'}


def test_a_wrong_sonic_val_pose_list_raises(tmp_path, tree):
    val_pose = make_pairs.derive_lines(SONIC_VAL, 'pose')
    val_pose[1] = 'logs/FOREIGN_BPL/Pose_2/P_9.npy logs/FOREIGN_BPL/Pose_2/P_9.npy'
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL,
                      val_pose_lines=val_pose)
    with pytest.raises(ValueError) as excinfo:
        make_pairs.main(['--root', tree, '--out', str(tmp_path / 'out'),
                         '--sonic-dir', sonic, '--n-train', '10', '--n-val', '2']
                        + SCENE_ARGS)
    assert 'line 2' in str(excinfo.value)


# --------------------------------------------------------------------------
# round 2: the verification reads the written files
# --------------------------------------------------------------------------

def rewrite(path, change):
    """Read a written list, change it and write it back."""
    lines = read_lines(path)
    change(lines)
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


@pytest.fixture
def written(tmp_path, tree):
    """Run the tool once. Returns (out, lists, sonic_lists, inventory)."""
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL)
    out = run_with_sonic(tmp_path, tree, sonic)
    inventory = make_pairs.Inventory.from_dir(tree)
    sonic_lists, _ = make_pairs.read_sonic_lists(sonic)
    pairs = {'train': make_pairs.generate(inventory, SCENES, 120, 5),
             'val': make_pairs.generate(inventory, SCENES, 30, 6)}
    return out, make_pairs.build_lists(sonic_lists, pairs), sonic_lists, inventory


def test_verify_accepts_the_files_it_wrote(written):
    """The fixture rebuilds the same lists, so the check must pass."""
    out, lists, sonic_lists, inventory = written
    report = make_pairs.verify(lists, sonic_lists, inventory, out)
    assert report['ok'] is True and report['reasons'] == []


def test_verify_finds_a_third_token(written):
    out, lists, sonic_lists, inventory = written
    rewrite(os.path.join(out, 'pairs.txt'),
            lambda lines: lines.__setitem__(3, lines[3] + ' logs/A_BPH_1/Sonar_0/S_1.npy'))
    report = make_pairs.verify(lists, sonic_lists, inventory, out)
    assert report['ok'] is False
    assert any('two paths' in reason for reason in report['reasons'])


def test_verify_finds_a_changed_sonic_line(written):
    """A changed SONIC line keeps the line count, so the count check is blind."""
    out, lists, sonic_lists, inventory = written
    rewrite(os.path.join(out, 'pairs.txt'),
            lambda lines: lines.__setitem__(0, lines[1]))
    report = make_pairs.verify(lists, sonic_lists, inventory, out)
    assert report['ok'] is False
    assert any('SONIC' in reason for reason in report['reasons'])


def test_verify_finds_swapped_pose_lines(written):
    out, lists, sonic_lists, inventory = written
    def swap(lines):
        lines[-1], lines[-2] = lines[-2], lines[-1]
    rewrite(os.path.join(out, 'pairs_pos.txt'), swap)
    report = make_pairs.verify(lists, sonic_lists, inventory, out)
    assert report['ok'] is False
    assert any('image list' in reason for reason in report['reasons'])


def test_verify_finds_a_missing_line(written):
    out, lists, sonic_lists, inventory = written
    rewrite(os.path.join(out, 'pairs_meta_val.txt'), lambda lines: lines.pop())
    report = make_pairs.verify(lists, sonic_lists, inventory, out)
    assert report['ok'] is False
    assert any('lines' in reason for reason in report['reasons'])


def test_verify_finds_a_pose_path_in_the_image_list(written):
    out, lists, sonic_lists, inventory = written
    rewrite(os.path.join(out, 'pairs_cross.txt'),
            lambda lines: lines.__setitem__(0, make_pairs.derive_lines(
                [lines[0]], 'pose')[0]))
    report = make_pairs.verify(lists, sonic_lists, inventory, out)
    assert report['ok'] is False
    assert any('image' in reason for reason in report['reasons'])


def test_cli_returns_2_when_a_written_file_is_wrong(tmp_path, tree, monkeypatch):
    """A bad write does not pass the verification."""
    correct = make_pairs.write_lines

    def broken(path, lines):
        if os.path.basename(path) == 'pairs.txt':
            lines = list(lines)
            lines[0] = lines[0] + ' logs/A_BPH_1/Sonar_0/S_1.npy'
        correct(path, lines)

    monkeypatch.setattr(make_pairs, 'write_lines', broken)
    sonic = sonic_dir(tmp_path / 'sonic', SONIC_TRAIN, SONIC_VAL)
    code = make_pairs.main(['--root', tree, '--out', str(tmp_path / 'out'),
                            '--sonic-dir', sonic, '--n-train', '40',
                            '--n-val', '10', '--verify'] + SCENE_ARGS)
    assert code == 2


# --------------------------------------------------------------------------
# round 2: provenance
# --------------------------------------------------------------------------

def test_manifest_holds_the_provenance(run_out):
    _, out = run_out
    manifest = read_manifest(out)
    with open(make_pairs.__file__, 'rb') as handle:
        assert manifest['script_sha256'] == hashlib.sha256(handle.read()).hexdigest()
    assert len(manifest['script_sha256']) == 64
    assert isinstance(manifest['git_dirty'], bool)
    assert manifest['numpy_version'] == np.__version__
    assert manifest['python_version'].startswith('3.')
    assert manifest['sonic_pose_lists'] == {'train': 'kept', 'val': 'derived'}


def test_manifest_holds_no_local_path(run_out):
    """The manifest keeps the names only, so you can publish it."""
    _, out = run_out
    inputs = read_manifest(out)['inputs']
    names = []
    for value in inputs.values():
        names.extend(value if isinstance(value, list) else [value])
    assert names
    for name in names:
        assert name is None or '/' not in name
    assert inputs['sonic_dir'] is not None and inputs['out'] is not None
