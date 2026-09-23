"""Build the C-SONIC pair lists.

C-SONIC trains on three parallel lists: image pairs, pose pairs and metadata
pairs. Line n of each list describes the same pair. This tool writes the six
training and validation lists.

The tool keeps the SONIC lists unchanged. It writes the SONIC lines first, in
their original order. It then adds new pairs that it draws from the C-SONIC
logs. A new pair holds two images of one scene. The two images can come from
one log or from two logs of that scene. Two logs of one scene show the same
trajectory at a different sonar frequency, at a different range, or both. A
cross-log pair with the same sonar and the same frame therefore shows one
position through two different sensors.

The tool writes the new pairs to their own files before it adds them to the
SONIC lines. It does not change the order of the lines. The dataloader mixes
the lines when it reads them.

Read the dataset from a directory with --root, from the zip files with --zips,
or from both. The tool reads the list of names of a zip file: no archive member
is opened.

Example:
    python make_pairs.py --zips /data/logs-zip/*.zip \\
        --sonic-dir /data/sonic-pairs --out /data/csonic-pairs --verify
"""
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import zipfile
from collections import Counter

import numpy as np

# The logs of one scene. All logs of a scene hold the same sonars, the same
# frames and the same poses.
DEFAULT_SCENES = {
    'ELC1': ['ELC1_BPH_1', 'ELC1_BPL_1'],
    'ELC2': ['ELC2_BPH_1', 'ELC2_BPL_1', 'ELC2_BPL_1_LR'],
    'RW': ['RW_BPL_1', 'RW_BPL_1_LR'],
}

# The three kinds of file, with their folder name and their file prefix.
KINDS = {'image': ('Sonar', 'S'), 'pose': ('Pose', 'P'), 'meta': ('Metadata', 'M')}
FOLDER_TO_KIND = {folder: kind for kind, (folder, _) in KINDS.items()}
ROOT_PREFIX = 'logs'

# The files that the tool writes. The *_cross* files hold the new pairs only.
CROSS_NAMES = {
    'train': ('pairs_cross.txt', 'pairs_pos_cross.txt', 'pairs_meta_cross.txt'),
    'val': ('pairs_cross_val.txt', 'pairs_pos_cross_val.txt',
            'pairs_meta_cross_val.txt'),
}
MERGED_NAMES = {
    'train': ('pairs.txt', 'pairs_pos.txt', 'pairs_meta.txt'),
    'val': ('pairs_val.txt', 'pairs_pos_val.txt', 'pairs_meta_val.txt'),
}
SPLITS = ('train', 'val')
LIST_KINDS = ('image', 'pose', 'meta')
ALL_NAMES = tuple(name for split in SPLITS
                  for names in (CROSS_NAMES[split], MERGED_NAMES[split])
                  for name in names)
MANIFEST_NAME = 'pairs_manifest.json'

# The SONIC lists that the tool reads. The release ships no validation pose
# list, so the tool derives that one.
SONIC_FILES = {'train': {'image': 'pairs.txt', 'pose': 'pairs_pos.txt'},
               'val': {'image': 'pairs_val.txt', 'pose': 'pairs_pos_val.txt'}}
# The number of draw rounds that may add no new pair before the tool stops.
IDLE_ROUNDS = 50


# --------------------------------------------------------------------------
# path convention
# --------------------------------------------------------------------------

def _make_path(kind, log, sonar, frame):
    """Build one dataset path. The path is relative and holds forward slashes."""
    folder, prefix = KINDS[kind]
    return '{}/{}/{}_{}/{}_{}.npy'.format(ROOT_PREFIX, log, folder, sonar,
                                          prefix, frame)


def image_path(log, sonar, frame):
    """Return the image path of one frame."""
    return _make_path('image', log, sonar, frame)


def parse_path(path):
    """Split a dataset path. Returns (kind, log, sonar, frame).

    The function reads the path components. A log name that holds 'S_' or
    'Sonar_' is therefore safe. A path that does not follow the convention
    raises ValueError.
    """
    parts = path.split('/')
    if len(parts) != 4 or parts[0] != ROOT_PREFIX or not parts[1]:
        raise ValueError('not a dataset path: {!r}'.format(path))
    log, folder, name = parts[1], parts[2], parts[3]
    if not name.endswith('.npy'):
        raise ValueError('not a .npy file: {!r}'.format(path))
    folder_name, _, sonar = folder.rpartition('_')
    prefix, _, frame = name[:-len('.npy')].partition('_')
    kind = FOLDER_TO_KIND.get(folder_name)
    if kind is None or KINDS[kind][1] != prefix:
        raise ValueError('the folder and the file do not match: {!r}'.format(path))
    if not sonar.isdigit() or not frame.isdigit():
        raise ValueError('the sonar or the frame is not a number: {!r}'.format(path))
    return kind, log, int(sonar), int(frame)


def parse_image_path(path):
    """Split an image path. Returns (log, sonar, frame)."""
    kind, log, sonar, frame = parse_path(path)
    if kind != 'image':
        raise ValueError('not an image path: {!r}'.format(path))
    return log, sonar, frame


def pose_path_from_image(path):
    """Return the pose path of an image path."""
    return _make_path('pose', *parse_image_path(path))


def meta_path_from_image(path):
    """Return the metadata path of an image path."""
    return _make_path('meta', *parse_image_path(path))


DERIVE = {'pose': pose_path_from_image, 'meta': meta_path_from_image}


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------

class Inventory(object):
    """The frames that the dataset holds, per log and per sonar."""

    def __init__(self, logs=None):
        # log -> sonar -> sorted numpy array of frame numbers
        self._logs = {} if logs is None else logs

    @property
    def logs(self):
        """The names of the logs, sorted."""
        return sorted(self._logs)

    def has_log(self, log):
        """Tell whether the inventory holds the log."""
        return log in self._logs

    def sonars(self, log):
        """The sonar numbers of one log, sorted."""
        return sorted(self._logs[log])

    def frames(self, log, sonar):
        """The frame numbers of one sonar, as a sorted array."""
        return self._logs[log][sonar]

    def has(self, path):
        """Tell whether the inventory holds one image, pose or metadata file."""
        _, log, sonar, frame = parse_path(path)
        frames = self._logs.get(log, {}).get(sonar)
        if frames is None:
            return False
        index = int(np.searchsorted(frames, frame))
        return index < len(frames) and int(frames[index]) == frame

    def __eq__(self, other):
        if not isinstance(other, Inventory):
            return NotImplemented
        if self.logs != other.logs:
            return False
        for log in self.logs:
            if self.sonars(log) != other.sonars(log):
                return False
            for sonar in self.sonars(log):
                if not np.array_equal(self.frames(log, sonar),
                                      other.frames(log, sonar)):
                    return False
        return True

    @staticmethod
    def merge(first, second):
        """Join two inventories. A log in both keeps the union of its frames."""
        logs = {}
        for source in (first, second):
            for log, sonars in source._logs.items():
                target = logs.setdefault(log, {})
                for sonar, frames in sonars.items():
                    if sonar in target:
                        target[sonar] = np.union1d(target[sonar], frames)
                    else:
                        target[sonar] = frames
        return Inventory(logs)

    @classmethod
    def _build(cls, entries, source):
        """Index the entries and check the three folders against each other."""
        found = {}
        for kind, log, sonar, frame in entries:
            found.setdefault(log, {}).setdefault(kind, {}) \
                 .setdefault(sonar, set()).add(frame)
        logs = {}
        for log in sorted(found):
            images = found[log].get('image', {})
            for kind in ('pose', 'meta'):
                other = found[log].get(kind, {})
                for sonar in sorted(set(images) | set(other)):
                    left = images.get(sonar, set())
                    right = other.get(sonar, set())
                    if left != right:
                        folder = KINDS[kind][0]
                        missing = sorted(left - right)
                        extra = sorted(right - left)
                        raise ValueError(
                            '{}: log {}: folder {}_{} does not match Sonar_{}: '
                            '{} frame(s) missing (first {}), {} frame(s) too many '
                            '(first {})'.format(
                                source, log, folder, sonar, sonar, len(missing),
                                missing[:1], len(extra), extra[:1]))
            logs[log] = {sonar: np.array(sorted(frames), dtype=np.int64)
                         for sonar, frames in images.items()}
        return cls(logs)

    @classmethod
    def from_dir(cls, root):
        """Read the inventory from <root>/logs/<LOG>/<folder>/<file>.npy."""
        base = os.path.join(root, ROOT_PREFIX)
        if not os.path.isdir(base):
            raise ValueError('no {} folder under {}'.format(ROOT_PREFIX, root))

        def entries():
            for log in sorted(os.listdir(base)):
                log_dir = os.path.join(base, log)
                if not os.path.isdir(log_dir):
                    continue
                for folder in sorted(os.listdir(log_dir)):
                    if not os.path.isdir(os.path.join(log_dir, folder)):
                        continue
                    name, _, index = folder.rpartition('_')
                    if name not in FOLDER_TO_KIND or not index.isdigit():
                        continue
                    for item in os.listdir(os.path.join(log_dir, folder)):
                        if item.endswith('.npy'):
                            yield parse_path('/'.join((ROOT_PREFIX, log, folder,
                                                       item)))
        return cls._build(entries(), root)

    @classmethod
    def from_zips(cls, paths):
        """Read the inventory from the list of names of each zip file.

        The tool does not open the members. The folder inside the zip must
        carry the name of the zip.
        """
        result = cls()
        for path in paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            with zipfile.ZipFile(path) as handle:
                names = handle.namelist()
            entries = []
            for name in names:
                parts = name.split('/')
                if len(parts) != 3 or not parts[2].endswith('.npy'):
                    continue
                if parts[0] != stem:
                    raise ValueError(
                        '{}: the folder {!r} inside the zip is not the name of '
                        'the zip ({!r})'.format(path, parts[0], stem))
                entries.append(parse_path('/'.join([ROOT_PREFIX] + parts)))
            if not entries:
                raise ValueError('{}: no .npy file in the dataset layout'.format(path))
            result = cls.merge(result, cls._build(entries, path))
        return result


# --------------------------------------------------------------------------
# scene table
# --------------------------------------------------------------------------

def parse_scene_args(values):
    """Read the --scene options. An empty list gives the default table."""
    if not values:
        return dict(DEFAULT_SCENES)
    scenes = {}
    for value in values:
        name, sep, logs = value.partition('=')
        if not sep or not name.strip() or not logs.strip():
            raise ValueError('--scene wants NAME=LOG,LOG,...: got {!r}'.format(value))
        scenes[name.strip()] = [log.strip() for log in logs.split(',') if log.strip()]
    return scenes


def check_scenes(inventory, scenes):
    """Check the scene table against the inventory.

    A log must be in one scene once only. Every log of a scene must be in the
    inventory. All logs of a scene must hold the same sonars and the same
    frames. All sonars of a scene must hold the same frames, because the
    generator draws one frame for two sonars.
    Returns {scene: (sonars, frames)} with both as sorted arrays.
    """
    scene_of = {}
    for scene in sorted(scenes):
        seen = set()
        for log in scenes[scene]:
            if log in seen:
                raise ValueError(
                    'scene {}: the log {} is in the scene twice'.format(scene, log))
            seen.add(log)
            if log in scene_of:
                raise ValueError('the log {} is in the scenes {} and {}'
                                 .format(log, scene_of[log], scene))
            scene_of[log] = scene
    table = {}
    for scene in sorted(scenes):
        logs = scenes[scene]
        if not logs:
            raise ValueError('scene {}: the scene holds no log'.format(scene))
        for log in logs:
            if not inventory.has_log(log):
                raise ValueError(
                    'scene {}: log {} is not in the inventory'.format(scene, log))
        first = logs[0]
        sonars = inventory.sonars(first)
        frames = inventory.frames(first, sonars[0])
        for log in logs:
            if inventory.sonars(log) != sonars:
                raise ValueError(
                    'scene {}: logs {} and {} hold different sonars: {} and {}'
                    .format(scene, first, log, sonars, inventory.sonars(log)))
            for sonar in sonars:
                if not np.array_equal(inventory.frames(log, sonar), frames):
                    raise ValueError(
                        'scene {}: log {}: Sonar_{} of {} and Sonar_{} of {} hold '
                        'different frames ({} and {})'.format(
                            scene, log, sonars[0], first, sonar, log,
                            len(frames), len(inventory.frames(log, sonar))))
        table[scene] = (np.array(sonars, dtype=np.int64), frames)
    return table


# --------------------------------------------------------------------------
# generator
# --------------------------------------------------------------------------

def allocate(total, weights):
    """Share a total over the weights. Largest-remainder keeps the sum exact."""
    if total < 0:
        raise ValueError('the total must not be negative: {}'.format(total))
    weights = [float(weight) for weight in weights]
    for weight in weights:
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('a weight must be a number of zero or more: {}'
                             .format(weight))
    if sum(weights) <= 0:
        raise ValueError('the weights must not be zero')
    exact = [total * weight / sum(weights) for weight in weights]
    parts = [int(value) for value in exact]
    order = sorted(range(len(weights)),
                   key=lambda i: (-(exact[i] - parts[i]), i))
    for index in order[:total - sum(parts)]:
        parts[index] += 1
    return parts


def frames_with_gap(frames, gap):
    """Count the frames f for which the log also holds the frame f + gap."""
    return int(np.intersect1d(frames, frames + gap, assume_unique=True).size)


def scene_capacity(n_logs, n_sonars, frames, gap_max):
    """Count the different pairs that one scene can give.

    Every ordered log pair meets every ordered sonar pair. Inside one log the
    gap is 1 to gap_max. Across two logs the gap is 0 to gap_max. A first frame
    counts only when the log also holds the second frame.
    """
    if gap_max < 1:
        raise ValueError('the largest gap must be 1 or more: {}'.format(gap_max))
    per_gap = [frames_with_gap(frames, gap) for gap in range(gap_max + 1)]
    inside = sum(per_gap[1:])
    across = sum(per_gap)
    return n_sonars ** 2 * (n_logs * inside + (n_logs ** 2 - n_logs) * across)


def _generate_scene(logs, sonars, frames, n_pairs, rng, gap_max):
    """Draw the pairs of one scene. Returns a list of (image, image)."""
    n_logs, n_sonars, n_frames = len(logs), len(sonars), len(frames)
    pairs = []
    seen = set()
    idle = 0
    while len(pairs) < n_pairs:
        found = len(pairs)
        size = int((n_pairs - len(pairs)) * 1.5) + 64
        first_log = rng.integers(0, n_logs, size)
        second_log = rng.integers(0, n_logs, size)
        first_sonar = rng.integers(0, n_sonars, size)
        second_sonar = rng.integers(0, n_sonars, size)
        first_frame = frames[rng.integers(0, n_frames, size)]
        # A pair inside one log needs a gap of one frame or more. A pair of two
        # logs can hold the same frame, which gives a pure cross-modal pair.
        gap = rng.integers(np.where(first_log == second_log, 1, 0), gap_max + 1)
        second_frame = first_frame + gap
        # Keep the draws whose second frame the log holds. The tool drops the
        # other draws and makes new ones.
        index = np.searchsorted(frames, second_frame)
        keep = index < n_frames
        good = np.flatnonzero(keep)
        good = good[frames[index[good]] == second_frame[good]]
        for row in good:
            pair = (image_path(logs[first_log[row]], sonars[first_sonar[row]],
                               first_frame[row]),
                    image_path(logs[second_log[row]], sonars[second_sonar[row]],
                               second_frame[row]))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
            if len(pairs) == n_pairs:
                break
        if len(pairs) > found:
            idle = 0
            continue
        idle += 1
        if idle >= IDLE_ROUNDS:
            raise RuntimeError(
                '{} rounds of draws added no new pair: the scene of the logs {} '
                'gives fewer than {} different pairs'.format(idle, logs, n_pairs))
    return pairs


def generate(inventory, scenes, n_pairs, seed, gap_max=9):
    """Draw n_pairs new image pairs. Returns a list of (image, image).

    Each scene gets a share of the pairs in proportion to its number of sonars
    times its number of frames. The same inputs and the same seed always give
    the same list. The tool does not change the order of the list.

    The tool draws a proposal, then keeps it or drops it: a proposal whose
    second frame lies beyond the end of the log is redrawn as a whole. A long
    gap is therefore a little more likely to be dropped than a short one. A
    pair inside one log holds a gap of 1 or more, a pair of two logs can hold
    the gap 0, so the two classes lose a different share. On the released data
    the shares of the log combinations and of the sonars move away from the
    uniform share by less than 0.03 % (0.015 % at most, see the task record).
    """
    if gap_max < 1:
        raise ValueError('the largest gap must be 1 or more: {}'.format(gap_max))
    if n_pairs < 0:
        raise ValueError('the number of pairs must not be negative: {}'
                         .format(n_pairs))
    table = check_scenes(inventory, scenes)
    names = sorted(scenes)
    parts = allocate(n_pairs, [len(table[name][0]) * len(table[name][1])
                               for name in names])
    rng = np.random.default_rng(seed)
    pairs = []
    for name, count in zip(names, parts):
        sonars, frames = table[name]
        capacity = scene_capacity(len(scenes[name]), len(sonars), frames, gap_max)
        if count > capacity:
            raise ValueError(
                'scene {}: {} pairs asked for, but the scene gives {} different '
                'pairs only'.format(name, count, capacity))
        pairs.extend(_generate_scene(scenes[name], sonars, frames, count, rng,
                                     gap_max))
    return pairs


def _band(log):
    """Return the frequency band of a log name, or an empty string."""
    if '_BPH_' in log:
        return 'BPH'
    if '_BPL_' in log:
        return 'BPL'
    return ''


def categorise(pair, scenes=None):
    """Name the category of one pair.

    The category comes from the two log names. The frequency differs when one
    name holds '_BPH_' and the other holds '_BPL_'. The range differs when one
    name ends with '_LR' and the other does not. A pair of one log, or of two
    logs that agree in both, is a 'same-log' pair. With a scene table, the
    function also checks that the two logs are in one scene.
    """
    first = parse_image_path(pair[0])[0]
    second = parse_image_path(pair[1])[0]
    if scenes is not None:
        if not any(first in logs and second in logs for logs in scenes.values()):
            raise ValueError(
                'the logs {} and {} are not in one scene'.format(first, second))
    frequency = {_band(first), _band(second)} == {'BPH', 'BPL'}
    distance = first.endswith('_LR') != second.endswith('_LR')
    if frequency and distance:
        return 'cross-frequency-range'
    if frequency:
        return 'cross-frequency'
    if distance:
        return 'cross-range'
    return 'same-log'


def pair_stats(pairs, scenes=None):
    """Count the categories, the zero-motion pairs and the frame gaps."""
    categories = Counter()
    gaps = Counter()
    zero_motion = 0
    for pair in pairs:
        log1, sonar1, frame1 = parse_image_path(pair[0])
        log2, sonar2, frame2 = parse_image_path(pair[1])
        categories[categorise(pair, scenes)] += 1
        gaps[frame2 - frame1] += 1
        if log1 != log2 and sonar1 == sonar2 and frame1 == frame2:
            zero_motion += 1
    return {'pairs': len(pairs),
            'categories': dict(sorted(categories.items())),
            'zero_motion': zero_motion,
            'gap_histogram': {str(gap): gaps[gap] for gap in sorted(gaps)}}


# --------------------------------------------------------------------------
# lists and files
# --------------------------------------------------------------------------

def split_line(line, source, lineno):
    """Split one list line into two paths. A different count is an error."""
    parts = line.split()
    if len(parts) != 2:
        raise ValueError('{}:{}: expected two paths, got {!r}'
                         .format(source, lineno, line))
    return parts


def pair_lines(pairs):
    """Turn the pairs into list lines."""
    return ['{} {}'.format(first, second) for first, second in pairs]


def derive_lines(lines, kind):
    """Turn image lines into pose lines or metadata lines."""
    convert = DERIVE[kind]
    return [' '.join(convert(path) for path in split_line(line, kind, lineno))
            for lineno, line in enumerate(lines, start=1)]


def read_sonic_list(path):
    """Read a SONIC pair list. Returns the lines without their line break."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for lineno, line in enumerate(lines, start=1):
        split_line(line, path, lineno)
    return lines


def check_sonic_pose_list(image_lines, pose_lines, source):
    """Check the SONIC pose list against the list we derive from the images."""
    if len(pose_lines) != len(image_lines):
        raise ValueError('{}: holds {} lines, the image list holds {}'
                         .format(source, len(pose_lines), len(image_lines)))
    expected = derive_lines(image_lines, 'pose')
    for lineno, (found, want) in enumerate(zip(pose_lines, expected), start=1):
        if found.split() != want.split():
            raise ValueError('{}: line {} is {!r}, the image list gives {!r}'
                             .format(source, lineno, found, want))


def empty_sonic_lists():
    """Return an empty SONIC prefix, one empty list per split and kind."""
    return {split: {kind: [] for kind in LIST_KINDS} for split in SPLITS}


def read_sonic_lists(sonic_dir):
    """Read the SONIC lists of a folder.

    Returns ({split: {kind: lines}}, {split: 'kept' or 'derived'}). The tool
    keeps the lines of the SONIC pose list as they are, so that list is
    necessary. The SONIC release ships no validation pose list. The tool
    derives that one from the validation image list and says so.
    """
    lists, source = {}, {}
    for split in SPLITS:
        images = read_sonic_list(os.path.join(sonic_dir,
                                              SONIC_FILES[split]['image']))
        pose_path = os.path.join(sonic_dir, SONIC_FILES[split]['pose'])
        if os.path.isfile(pose_path):
            poses = read_sonic_list(pose_path)
            check_sonic_pose_list(images, poses, pose_path)
            source[split] = 'kept'
        elif split == 'train':
            raise FileNotFoundError(
                'the SONIC pose list is necessary: {}'.format(pose_path))
        else:
            poses = derive_lines(images, 'pose')
            source[split] = 'derived'
        lists[split] = {'image': images, 'pose': poses,
                        'meta': derive_lines(images, 'meta')}
    return lists, source


def build_lists(sonic_lists, new_pairs):
    """Build the twelve lists. Returns {file name: lines}.

    The SONIC lines come first, then the new lines. The tool keeps the order
    of both parts. The metadata lines are always derived from the image lines.
    """
    lists = {}
    for split in SPLITS:
        cross = pair_lines(new_pairs[split])
        for kind, cross_name, merged_name in zip(LIST_KINDS, CROSS_NAMES[split],
                                                 MERGED_NAMES[split]):
            new = cross if kind == 'image' else derive_lines(cross, kind)
            lists[cross_name] = new
            lists[merged_name] = sonic_lists[split][kind] + new
    return lists


def write_lines(path, lines):
    """Write one list. The file always ends with a line break."""
    with open(path, 'w') as handle:
        for line in lines:
            handle.write(line + '\n')


def file_digest(path, digest):
    """Read a file and return its sum, in hexadecimal."""
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def file_md5(path):
    """Return the md5 sum of a file."""
    return file_digest(path, hashlib.md5())


def file_sha256(path):
    """Return the sha256 sum of a file."""
    return file_digest(path, hashlib.sha256())


def per_log_line_counts(lines, column=0):
    """Count the lines per log, from one column of an image list."""
    counts = Counter()
    for lineno, line in enumerate(lines, start=1):
        path = split_line(line, 'list', lineno)[column]
        try:
            counts[parse_path(path)[1]] += 1
        except ValueError:
            counts['(malformed)'] += 1
    return dict(sorted(counts.items()))


LIST_KIND_OF = {name: kind
                for split in SPLITS
                for kind, cross_name, merged_name
                in zip(LIST_KINDS, CROSS_NAMES[split], MERGED_NAMES[split])
                for name in (cross_name, merged_name)}
MAX_REASONS = 40


def verify(lists, sonic_lists, inventory, out_dir=None):
    """Check the twelve lists closely. Returns the report.

    With out_dir the function reads the twelve written files, not the lists in
    memory: a bad write must not pass. The function checks that every line
    holds two paths of the kind of the file, that line n of the pose list and
    of the metadata list belongs to line n of the image list, that the SONIC
    lines come first and unchanged, that the new lines come after them and
    equal the *_cross* file byte for byte, that the number of lines is right,
    and that the inventory holds every file. A log that the inventory does not
    hold gives a 'not verifiable' report: the dataset of that log is somewhere
    else. A file that is missing from a log that the inventory holds is an
    error.
    """
    reasons = []

    def note(reason):
        """Write down one reason, up to the limit."""
        if len(reasons) < MAX_REASONS:
            reasons.append(reason)

    content = {}
    for name in ALL_NAMES:
        if out_dir is None:
            content[name] = lists[name]
        else:
            with open(os.path.join(out_dir, name)) as handle:
                content[name] = handle.read().splitlines()

    # Every line holds two paths, and both paths are of the kind of the file.
    good = {}
    for name in ALL_NAMES:
        want = LIST_KIND_OF[name]
        good[name] = True
        for lineno, line in enumerate(content[name], start=1):
            paths = line.split()
            if len(paths) != 2:
                note('{}:{}: expected two paths, got {!r}'.format(name, lineno, line))
                good[name] = False
                break
            for path in paths:
                try:
                    kind = parse_path(path)[0]
                except ValueError as error:
                    note('{}:{}: {}'.format(name, lineno, error))
                    good[name] = False
                    break
                if kind != want:
                    note('{}:{}: {} is not a path of the kind {}'
                         .format(name, lineno, path, want))
                    good[name] = False
                    break
            if not good[name]:
                break

    counts_ok = True
    for split in SPLITS:
        n_sonic = len(sonic_lists[split]['image'])
        n_new = len(lists[CROSS_NAMES[split][0]])
        for name in CROSS_NAMES[split]:
            if len(content[name]) != n_new:
                note('{}: holds {} lines, expected {} new lines'
                     .format(name, len(content[name]), n_new))
                counts_ok = False
        for name in MERGED_NAMES[split]:
            if len(content[name]) != n_sonic + n_new:
                note('{}: holds {} lines, expected {} SONIC lines and {} new '
                     'lines'.format(name, len(content[name]), n_sonic, n_new))
                counts_ok = False

        # The SONIC lines come first and unchanged. The new lines follow them.
        for kind, cross_name, merged_name in zip(LIST_KINDS, CROSS_NAMES[split],
                                                 MERGED_NAMES[split]):
            prefix = sonic_lists[split][kind]
            for lineno, (found, want) in enumerate(
                    zip(content[merged_name], prefix), start=1):
                if found != want:
                    note('{}: line {} is not the SONIC line: {!r} instead of {!r}'
                         .format(merged_name, lineno, found, want))
                    break
            if content[merged_name][len(prefix):] != content[cross_name]:
                note('{}: the lines after the SONIC lines are not the lines of {}'
                     .format(merged_name, cross_name))

        # Line n of the pose list and of the metadata list belongs to line n of
        # the image list. A kept SONIC line may hold other whitespace, so the
        # tool compares the paths of those lines, not the line itself.
        images = content[MERGED_NAMES[split][0]]
        cross_images = content[CROSS_NAMES[split][0]]
        for kind, cross_name, merged_name in zip(LIST_KINDS, CROSS_NAMES[split],
                                                 MERGED_NAMES[split]):
            if kind == 'image':
                continue
            convert = DERIVE[kind]
            for name, source_lines, n_verbatim in ((merged_name, images, n_sonic),
                                                   (cross_name, cross_images, 0)):
                if not good[name]:
                    continue
                for lineno, (found, image_line) in enumerate(
                        zip(content[name], source_lines), start=1):
                    try:
                        want = ' '.join(convert(path) for path in image_line.split())
                    except ValueError:
                        continue
                    same = (found.split() == want.split() if lineno <= n_verbatim
                            else found == want)
                    if not same:
                        note('{}: line {} does not match the image list: {!r} '
                             'instead of {!r}'.format(name, lineno, found, want))
                        break

    # The end of a merged file equals the matching *_cross* file byte for byte.
    tail_ok = None
    if out_dir is not None:
        tail_ok = True
        for split in SPLITS:
            for merged_name, cross_name in zip(MERGED_NAMES[split],
                                               CROSS_NAMES[split]):
                cross_path = os.path.join(out_dir, cross_name)
                merged_path = os.path.join(out_dir, merged_name)
                size = os.path.getsize(cross_path)
                with open(cross_path, 'rb') as handle:
                    cross = handle.read()
                with open(merged_path, 'rb') as handle:
                    handle.seek(max(os.path.getsize(merged_path) - size, 0))
                    if handle.read() != cross:
                        tail_ok = False
                        note('{}: the end of the file is not {} byte for byte'
                             .format(merged_name, cross_name))

    # The inventory holds every file that the lists name.
    paths = set()
    for name in ALL_NAMES:
        for line in content[name]:
            paths.update(line.split())
    missing, malformed, unknown, verified = [], [], set(), set()
    for path in sorted(paths):
        try:
            log = parse_path(path)[1]
        except ValueError:
            malformed.append(path)
            continue
        if not inventory.has_log(log):
            unknown.add(log)
        elif inventory.has(path):
            verified.add(log)
        else:
            missing.append(path)
    if missing:
        note('{} file(s) of a log of the inventory are missing, the first is {}'
             .format(len(missing), missing[0]))
    if malformed:
        note('{} path(s) do not follow the convention, the first is {}'
             .format(len(malformed), malformed[0]))

    lines_per_log = Counter()
    for split in SPLITS:
        for line in content[MERGED_NAMES[split][0]]:
            logs = set()
            for path in line.split():
                try:
                    logs.add(parse_path(path)[1])
                except ValueError:
                    pass
            for log in logs:
                lines_per_log[log] += 1

    return {'checked_paths': len(paths),
            'verified_logs': sorted(verified),
            'not_verifiable': {log: lines_per_log[log] for log in sorted(unknown)},
            'missing_count': len(missing),
            'missing_examples': missing[:5],
            'malformed_count': len(malformed),
            'malformed_examples': malformed[:5],
            'tail_ok': tail_ok,
            'line_counts': {name: len(content[name]) for name in ALL_NAMES},
            'line_counts_ok': counts_ok,
            'reasons': reasons,
            'ok': not reasons}


def git_sha():
    """Return the commit of the repository, or 'unknown'."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=20)
    except (OSError, subprocess.SubprocessError):
        return 'unknown'
    if result.returncode != 0:
        return 'unknown'
    return result.stdout.decode().strip() or 'unknown'


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def git_dirty():
    """Tell whether the repository holds changes. Read-only.

    The answer is False when the tool cannot ask git. The git sum is 'unknown'
    in that case, so the manifest stays clear.
    """
    try:
        result = subprocess.run(['git', 'status', '--porcelain'],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def name_only(path):
    """Return the name of a path, without the folders.

    The manifest keeps the names only. You can publish it as it is.
    """
    if not path:
        return None
    return os.path.basename(os.path.normpath(path))


def build_parser():
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description='Build the C-SONIC image, pose and metadata pair lists.')
    parser.add_argument('--root', help='dataset root that holds the logs folder')
    parser.add_argument('--zips', nargs='+', default=[], metavar='PATH',
                        help='log zip files; the tool reads the names only')
    parser.add_argument('--sonic-dir',
                        help='folder of the SONIC pairs.txt and pairs_val.txt')
    parser.add_argument('--out', help='output folder (default: <root>/logs)')
    parser.add_argument('--n-train', type=int, default=255122,
                        help='number of new training pairs')
    parser.add_argument('--n-val', type=int, default=27578,
                        help='number of new validation pairs')
    parser.add_argument('--seed', type=int, default=2024,
                        help='seed of the training draw; the validation draw '
                             'uses this seed plus one')
    parser.add_argument('--gap-max', type=int, default=9,
                        help='largest frame gap inside a pair')
    parser.add_argument('--scene', action='append', default=[],
                        metavar='NAME=LOG,LOG', help='replace the scene table')
    parser.add_argument('--verify', action='store_true',
                        help='check every path of every list')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the counts and write nothing')
    return parser


def main(argv=None):
    """Run the tool. Returns the exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.root and not args.zips:
        parser.error('give --root, --zips, or both')
    out_dir = args.out or (os.path.join(args.root, ROOT_PREFIX) if args.root else None)
    if out_dir is None:
        parser.error('--out is needed when you give --zips only')
    if args.sonic_dir and os.path.abspath(out_dir) == os.path.abspath(args.sonic_dir):
        raise ValueError('--out must not be the SONIC folder: it would '
                         'overwrite the SONIC lists')

    scenes = parse_scene_args(args.scene)
    sources = []
    if args.root:
        sources.append(Inventory.from_dir(args.root))
    if args.zips:
        sources.append(Inventory.from_zips(args.zips))
    inventory = sources[0] if len(sources) == 1 else Inventory.merge(*sources)
    print('inventory: {} log(s): {}'.format(len(inventory.logs),
                                            ', '.join(inventory.logs)))

    new_pairs = {'train': generate(inventory, scenes, args.n_train, args.seed,
                                   args.gap_max),
                 'val': generate(inventory, scenes, args.n_val, args.seed + 1,
                                 args.gap_max)}
    sonic_lists = empty_sonic_lists()
    pose_source = {split: 'none' for split in SPLITS}
    if args.sonic_dir:
        sonic_lists, pose_source = read_sonic_lists(args.sonic_dir)
        for split in SPLITS:
            print('SONIC {}: {} line(s), the pose list is {}'.format(
                split, len(sonic_lists[split]['image']), pose_source[split]))

    lists = build_lists(sonic_lists, new_pairs)
    n_sonic = {split: len(sonic_lists[split]['image']) for split in SPLITS}

    manifest = {
        'seed': args.seed,
        'val_seed': args.seed + 1,
        'gap_max': args.gap_max,
        'git_sha': git_sha(),
        'git_dirty': git_dirty(),
        'script_sha256': file_sha256(os.path.abspath(__file__)),
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
        'sonic_pose_lists': pose_source,
        'inputs': {'root': name_only(args.root),
                   'zips': [name_only(path) for path in args.zips],
                   'sonic_dir': name_only(args.sonic_dir),
                   'out': name_only(out_dir)},
        'scenes': scenes,
        'counts': {split: {'sonic': n_sonic[split],
                           'new': len(new_pairs[split]),
                           'total': n_sonic[split] + len(new_pairs[split])}
                   for split in SPLITS},
        'new_pairs': {split: pair_stats(new_pairs[split], scenes)
                      for split in SPLITS},
        'per_log_lines': {split: per_log_line_counts(lists[MERGED_NAMES[split][0]])
                          for split in SPLITS},
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print('dry run: no file written')
        return 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for name in ALL_NAMES:
        write_lines(os.path.join(out_dir, name), lists[name])
    manifest['md5'] = {name: file_md5(os.path.join(out_dir, name))
                       for name in ALL_NAMES}
    manifest['verify'] = (verify(lists, sonic_lists, inventory, out_dir)
                          if args.verify else None)

    with open(os.path.join(out_dir, MANIFEST_NAME), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')

    for split in SPLITS:
        counts = manifest['counts'][split]
        print('{}: {} SONIC line(s) + {} new line(s) = {}'.format(
            split, counts['sonic'], counts['new'], counts['total']))
    report = manifest['verify']
    if report is not None:
        print('verify: {} path(s) checked, {} missing, {} malformed'.format(
            report['checked_paths'], report['missing_count'],
            report['malformed_count']))
        if report['not_verifiable']:
            print('verify: not verifiable (log not in the inventory): {}'.format(
                ', '.join('{} ({} lines)'.format(log, count)
                          for log, count in report['not_verifiable'].items())))
        if not report['ok']:
            for reason in report['reasons'][:10]:
                print('verify: {}'.format(reason), file=sys.stderr)
            for path in report['missing_examples'] + report['malformed_examples']:
                print('verify: bad path: {}'.format(path), file=sys.stderr)
            print('verify: FAILED', file=sys.stderr)
            return 2
        print('verify: ok')
    print('wrote {} list(s) and the manifest to {}'.format(len(ALL_NAMES), out_dir))
    return 0


if __name__ == '__main__':
    sys.exit(main())
