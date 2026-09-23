"""Extract a small sample of the C-SONIC dataset from the log zip files.

The sample holds a few frames of one scene, with the six pair lists that the
dataloader and the notebook read. The tool reads single members of an archive:
it never unpacks one. A 36 GB zip file therefore costs a few megabytes of
reading.

The tool writes the layout of the release:

    <out>/logs/<LOG>/Sonar_<n>/S_<f>.npy
    <out>/logs/<LOG>/Pose_<n>/P_<f>.npy
    <out>/logs/<LOG>/Metadata_<n>/M_<f>.npy
    <out>/logs/pairs.txt and the five other lists

One log of the scene is the anchor, and every pair holds an image of the
anchor. The tool writes the cross-log pairs first: they show one position
through two sensors, or through one sensor a few frames apart. It writes the
same-log pairs of the anchor after them: they show the motion of one sensor.

The high-frequency image comes first in every pair that holds one
high-frequency (_BPH_) image and one other image. The high-frequency view is
the narrower view, so most of its query points are inside the view of the
other image. The model matches the points of image 1, and the notebook shows
image 1 on the left. In all other pairs the anchor image comes first. Give
<out> to the notebook as args.datadir.

A folder can hold more than one scene. Run the tool again with --append: it
reads the lists that are already there, keeps those lines first and in their
order, and adds the new lines after them. A pair that is already there, in
either order, is not written twice, and the tool reads from the archives only
the files of the pairs that are new. The same run twice therefore changes
nothing at all.
Without --append the tool refuses to write into a folder that already holds
the lists, because that would drop the pairs of the earlier run.

The tool reads the manifest and checks the folder against it before it writes
the first file, so a folder that does not match its manifest stays as it is.

The manifest grows with the folder. The key 'runs' holds one record per run,
so the source of every scene stays known. The keys 'md5', 'files' and
'total_bytes' describe the .npy data files of the folder; the preview images
are named under 'preview'. The other keys describe the latest run.

Example:
    python extract_sample.py --zips /data/logs-zip/ELC2_BPH_1.zip \\
        /data/logs-zip/ELC2_BPL_1.zip /data/logs-zip/ELC2_BPL_1_LR.zip \\
        --out samples/csonic_sample --preview
    python extract_sample.py --zips /data/logs-zip/ELC1_BPH_1.zip \\
        /data/logs-zip/ELC1_BPL_1.zip --scene ELC1 --frames 1200 1204 1208 \\
        --max-gap 4 --out samples/csonic_sample --preview --append
"""
import argparse
import io
import json
import os
import sys
import zipfile

import numpy as np

import make_pairs

MANIFEST_NAME = 'sample_manifest.json'
PREVIEW_DIR = 'preview'
DEFAULT_SCENE = 'ELC2'
DEFAULT_SONAR = 0
DEFAULT_FRAMES = [1000, 1005, 1010]
DEFAULT_MAX_GAP = 5
# The keys of the manifest that describe the plan of one run. The manifest
# keeps one record of these keys per run under 'runs'.
RUN_KEYS = ('anchor', 'frames', 'git_sha', 'logs', 'max_gap', 'scene',
            'script_sha256', 'sonar', 'source_zips')


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def unique(values):
    """Return the values without repeats. The first order stays."""
    seen, result = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def is_high_frequency(log):
    """Tell whether a log is a high-frequency log. Its name holds '_BPH_'."""
    return '_BPH_' in log


def choose_anchor(logs):
    """Return the anchor: the log that every pair holds.

    The anchor is the first low-frequency log at the normal range. The name of
    such a log holds '_BPL_' and does not end with '_LR'.
    """
    for log in logs:
        if '_BPL_' in log and not log.endswith('_LR'):
            return log
    raise ValueError('none of the logs {} is a _BPL_ log at the normal range: '
                     'name the anchor with --anchor'.format(', '.join(logs)))


def build_pairs(logs, anchor, sonar, frames, max_gap):
    """Build the image pairs of the sample. Returns [(first, second)].

    The tool walks the other logs in the order of the scene and pairs every
    frame of the anchor with every frame of that log. It then pairs the frames
    of the anchor with each other, the earlier frame first. A pair needs a
    frame gap of max_gap or less. The order is always the same.

    When the other log is high frequency and the anchor is not, the image of
    the other log comes first. In all other pairs the anchor image comes first.
    """
    if max_gap < 0:
        raise ValueError('--max-gap must not be negative: {}'.format(max_gap))
    if anchor not in logs:
        raise ValueError('the anchor {} is not one of the logs {}'
                         .format(anchor, ', '.join(logs)))
    pairs = []
    for other in logs:
        if other == anchor:
            continue
        swap = is_high_frequency(other) and not is_high_frequency(anchor)
        for first in frames:
            for second in frames:
                if abs(first - second) <= max_gap:
                    pair = (make_pairs.image_path(anchor, sonar, first),
                            make_pairs.image_path(other, sonar, second))
                    pairs.append(pair[::-1] if swap else pair)
    for first in frames:
        for second in frames:
            if 0 < second - first <= max_gap:
                pairs.append((make_pairs.image_path(anchor, sonar, first),
                              make_pairs.image_path(anchor, sonar, second)))
    return pairs


def check_available(inventory, images):
    """Check the plan against the archives. Names the first file it misses.

    The inventory holds the three kinds of file of a log together, so a check
    of the image also covers its pose and its metadata.
    """
    for path in images:
        if not inventory.has(path):
            raise ValueError('the zip files hold no {}'.format(path))


# --------------------------------------------------------------------------
# the archives
# --------------------------------------------------------------------------

def member_name(path):
    """Turn a dataset path into the name of the member of the zip file.

    A zip holds <LOG>/<folder>/<file>. A dataset path carries the name of the
    logs folder in front of that name.
    """
    prefix = make_pairs.ROOT_PREFIX + '/'
    if not path.startswith(prefix):
        raise ValueError('not a dataset path: {!r}'.format(path))
    return path[len(prefix):]


def zips_by_log(paths):
    """Map the name of a log to its zip file. The name of the zip names the log."""
    result = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in result:
            raise ValueError('two zip files carry the name {}'.format(stem))
        result[stem] = path
    return result


def sibling_paths(image):
    """Return the image, the pose and the metadata path of one frame."""
    return (image, make_pairs.pose_path_from_image(image),
            make_pairs.meta_path_from_image(image))


def preview_name(image):
    """Return the file name of the preview of one image."""
    log, sonar, frame = make_pairs.parse_image_path(image)
    return '{}_S{}_F{}.png'.format(log, sonar, frame)


def write_preview(path, data):
    """Write one PNG of an image. The grey scale runs from the low to the high.

    The tool imports Pillow here, so that a run without --preview needs no
    image library.
    """
    from PIL import Image

    array = np.asarray(np.load(io.BytesIO(data), allow_pickle=True),
                       dtype=np.float64)
    # The dataloader flips the image, because the sensor sits at the bottom of
    # the view it shows. The preview must show the same picture.
    array = np.flip(array)
    low, high = float(array.min()), float(array.max())
    scaled = (array - low) / (high - low) if high > low else np.zeros_like(array)
    Image.fromarray(np.round(scaled * 255).astype(np.uint8)).save(path)


def extract(zip_of_log, images, out_dir, preview=False):
    """Copy the files of every image. Returns (written paths, preview names).

    The tool opens each zip file once and reads the named members only. It
    writes the bytes of a member as they are, so the copy and the member are
    the same file.
    """
    groups = {}
    for image in images:
        groups.setdefault(make_pairs.parse_image_path(image)[0], []).append(image)

    written, previews = [], []
    preview_dir = os.path.join(out_dir, PREVIEW_DIR)
    for log, group in groups.items():
        with zipfile.ZipFile(zip_of_log[log]) as handle:
            for image in group:
                data = [handle.read(member_name(path))
                        for path in sibling_paths(image)]
                for path, block in zip(sibling_paths(image), data):
                    target = os.path.join(out_dir, *path.split('/'))
                    folder = os.path.dirname(target)
                    if not os.path.isdir(folder):
                        os.makedirs(folder)
                    with open(target, 'wb') as out:
                        out.write(block)
                    written.append(path)
                if preview:
                    if not os.path.isdir(preview_dir):
                        os.makedirs(preview_dir)
                    name = preview_name(image)
                    write_preview(os.path.join(preview_dir, name), data[0])
                    previews.append(name)
    return written, previews


# --------------------------------------------------------------------------
# the lists
# --------------------------------------------------------------------------

def write_lists(list_dir, lines):
    """Write the six pair lists. The training lists and the validation lists
    hold the same pairs: the notebook reads the validation lists, a training
    run reads the others."""
    for split in make_pairs.SPLITS:
        for kind, name in zip(make_pairs.LIST_KINDS,
                              make_pairs.MERGED_NAMES[split]):
            content = lines if kind == 'image' else make_pairs.derive_lines(lines,
                                                                            kind)
            make_pairs.write_lines(os.path.join(list_dir, name), content)


def list_paths(list_dir):
    """Return the paths of the six pair lists, the image lists first."""
    return [os.path.join(list_dir, name) for split in make_pairs.SPLITS
            for name in make_pairs.MERGED_NAMES[split]]


def existing_lists(list_dir):
    """Return the pair lists that the folder already holds."""
    return [path for path in list_paths(list_dir) if os.path.isfile(path)]


def read_existing_lines(list_dir):
    """Read the image list of an earlier run. Returns [] when there is none.

    The folder must hold all six lists or none of them. The training image list
    and the validation image list must agree, because this tool writes the same
    pairs to both. A folder that breaks one of these rules is not a folder that
    the tool made, so the tool stops instead of guessing.
    """
    if not existing_lists(list_dir):
        return []
    for path in list_paths(list_dir):
        if not os.path.isfile(path):
            raise ValueError('the folder holds some of the six lists only: {} '
                             'is missing'.format(path))
    lines = {}
    for split in make_pairs.SPLITS:
        name = make_pairs.MERGED_NAMES[split][0]
        lines[split] = make_pairs.read_sonic_list(os.path.join(list_dir, name))
    if lines['train'] != lines['val']:
        raise ValueError('{} and {} hold different pairs: the tool cannot add '
                         'to these lists'.format(
                             make_pairs.MERGED_NAMES['train'][0],
                             make_pairs.MERGED_NAMES['val'][0]))
    return lines['train']


def merge_lines(old, new):
    """Add the new lines to the old ones. Returns the merged list.

    The old lines stay first and keep their order. A new line whose pair is
    already there is not added a second time, in either order: a folder made
    by an earlier version put the anchor image first in every pair, and the
    same pair must not come back with its two images swapped.
    """
    seen = set(pair_key(line) for line in old)
    merged = list(old)
    for line in new:
        key = pair_key(line)
        if key not in seen:
            seen.add(key)
            merged.append(line)
    return merged


def pair_key(line):
    """Return the two images of one list line, in a fixed order."""
    return tuple(sorted(line.split()))


# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

def read_manifest(path):
    """Read the manifest of an earlier run. Returns {} when there is none."""
    if not os.path.isfile(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def run_record(plan):
    """Pull the keys of one run out of a manifest or a plan."""
    return {key: plan.get(key) for key in RUN_KEYS}


def earlier_runs(manifest):
    """Return the run records of a manifest.

    A manifest that the first release wrote holds no 'runs' key. Its own keys
    describe that one run, so the tool makes the record from them.
    """
    if not manifest:
        return []
    runs = manifest.get('runs')
    if runs is None:
        return [run_record(manifest)]
    return list(runs)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def build_parser():
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description='Extract a small C-SONIC sample from the log zip files.')
    parser.add_argument('--zips', nargs='+', required=True, metavar='PATH',
                        help='the log zip files; the name of a zip names its log')
    parser.add_argument('--out', required=True, metavar='DIR',
                        help='the root of the sample; the tool writes <out>/logs')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--scene', metavar='NAME',
                       help='a scene of the release table (default: {})'
                            .format(DEFAULT_SCENE))
    group.add_argument('--logs', nargs='+', metavar='LOG',
                       help='the logs of the scene, in the order of the pairs')
    parser.add_argument('--anchor', metavar='LOG',
                        help='the log that every pair holds (default: the '
                             'first _BPL_ log at the normal range); a pair '
                             'starts with its _BPH_ image if it holds one, '
                             'else with the anchor image')
    parser.add_argument('--sonar', type=int, default=DEFAULT_SONAR,
                        help='the sonar to take the frames from')
    parser.add_argument('--frames', type=int, nargs='+', default=DEFAULT_FRAMES,
                        metavar='N', help='the frames to take')
    parser.add_argument('--max-gap', type=int, default=DEFAULT_MAX_GAP,
                        metavar='N', help='the largest frame gap inside a pair')
    parser.add_argument('--append', action='store_true',
                        help='add the new pairs to the lists that <out> already '
                             'holds; without it the tool refuses to write into '
                             'a folder that holds the lists')
    parser.add_argument('--preview', action='store_true',
                        help='also write a PNG of every image under <out>/preview')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the plan and write nothing')
    return parser


def main(argv=None):
    """Run the tool. Returns the exit code."""
    args = build_parser().parse_args(argv)

    if args.logs:
        scene, logs = None, list(args.logs)
    else:
        scene = args.scene or DEFAULT_SCENE
        if scene not in make_pairs.DEFAULT_SCENES:
            raise ValueError('unknown scene {!r}: the table holds {}'.format(
                scene, ', '.join(sorted(make_pairs.DEFAULT_SCENES))))
        logs = list(make_pairs.DEFAULT_SCENES[scene])
    if len(set(logs)) != len(logs):
        raise ValueError('a log is in the scene twice: {}'.format(', '.join(logs)))
    anchor = args.anchor or choose_anchor(logs)
    frames = sorted(unique(args.frames))

    pairs = build_pairs(logs, anchor, args.sonar, frames, args.max_gap)
    lines = make_pairs.pair_lines(pairs)
    images = unique([path for pair in pairs for path in pair])

    zip_of_log = zips_by_log(args.zips)
    inventory = make_pairs.Inventory.from_zips(args.zips)
    check_available(inventory, images)
    for log in logs:
        if log not in zip_of_log:
            raise ValueError('no zip file carries the name of the log {}'.format(log))

    out_dir = args.out
    list_dir = os.path.join(out_dir, make_pairs.ROOT_PREFIX)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)

    # Every check of the folder runs before the first write. A run that stops
    # here leaves the folder as it was.
    found = existing_lists(list_dir)
    if found and not args.append:
        raise ValueError('{} is already there: give --append to add the new '
                         'pairs to the lists of the earlier run, or give '
                         'another --out'.format(found[0]))
    old_lines, old_manifest = [], {}
    if args.append and found:
        old_lines = read_existing_lines(list_dir)
        if not os.path.isfile(manifest_path):
            raise ValueError('the folder holds the lists of an earlier run but '
                             'no {}: the tool cannot tell which data files '
                             'belong to them'.format(MANIFEST_NAME))
        old_manifest = read_manifest(manifest_path)
    old_files = list(old_manifest.get('md5', {}))
    for path in old_files:
        if not os.path.isfile(os.path.join(out_dir, *path.split('/'))):
            raise ValueError('the manifest of an earlier run names {}, which is '
                             'not in the folder: the tool writes nothing'
                             .format(path))

    merged = merge_lines(old_lines, lines)
    # Only the pairs that the lists do not hold yet need their files. A run
    # that adds no pair therefore reads and writes nothing.
    new_lines = merged[len(old_lines):]
    to_extract = unique([path for line in new_lines for path in line.split()])
    known = unique(old_files + [path for image in to_extract
                                for path in sibling_paths(image)])

    if args.dry_run:
        print('scene: {}, anchor: {}, sonar: {}, frames: {}, largest gap: {}'
              .format(scene or ', '.join(logs), anchor, args.sonar,
                      ', '.join(str(frame) for frame in frames), args.max_gap))
        for line in lines:
            print('pair: {}'.format(line))
        print('dry run: {} pair(s), {} image(s), {} data file(s), nothing written'
              .format(len(pairs), len(to_extract), 3 * len(to_extract)))
        if old_lines:
            print('dry run: the lists would hold {} line(s): {} from earlier '
                  'runs and {} new'.format(len(merged), len(old_lines),
                                           len(new_lines)))
        return 0

    if old_lines and not new_lines:
        print('0 data file(s) written: the lists already hold every one of the '
              '{} pair(s) of this run, so the folder is unchanged'
              .format(len(pairs)))
        print('--datadir {}'.format(os.path.abspath(out_dir)))
        return 0

    if not os.path.isdir(list_dir):
        os.makedirs(list_dir)
    written, previews = extract(zip_of_log, to_extract, out_dir, args.preview)
    write_lists(list_dir, merged)

    # The manifest describes the folder, not the run: 'md5', 'files' and
    # 'total_bytes' cover every .npy data file that is there, and the tool sums
    # them again on every run. The previews are named under 'preview'.
    plan = {'anchor': anchor,
            'frames': frames,
            'git_sha': make_pairs.git_sha(),
            'logs': logs,
            'max_gap': args.max_gap,
            'scene': scene,
            'script_sha256': make_pairs.file_sha256(os.path.abspath(__file__)),
            'sonar': args.sonar,
            'source_zips': [make_pairs.name_only(path) for path in args.zips]}
    manifest = dict(plan)
    manifest.update({
        'files': len(known),
        'git_dirty': make_pairs.git_dirty(),
        'md5': {path: make_pairs.file_md5(os.path.join(out_dir, *path.split('/')))
                for path in known},
        'pairs': [line.split() for line in merged],
        'preview': unique(list(old_manifest.get('preview', [])) + previews),
        'runs': earlier_runs(old_manifest) + [run_record(plan)],
        'total_bytes': sum(os.path.getsize(os.path.join(out_dir, *path.split('/')))
                           for path in known),
    })
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')

    print('{} pair(s) of {} log(s), anchor {}, sonar {}, frames {}'.format(
        len(pairs), len(logs), anchor, args.sonar,
        ', '.join(str(frame) for frame in frames)))
    print('{} line(s) in each of the {} lists: {} from earlier runs, {} new'
          .format(len(merged),
                  len(make_pairs.MERGED_NAMES['train']) * len(make_pairs.SPLITS),
                  len(old_lines), len(merged) - len(old_lines)))
    print('{} data file(s) written, {} data file(s) in all, {:.1f} MB under {}'
          .format(len(written), manifest['files'],
                  manifest['total_bytes'] / 1e6, out_dir))
    if previews:
        print('{} preview image(s) written, {} in all, under {}'.format(
            len(previews), len(manifest['preview']),
            os.path.join(out_dir, PREVIEW_DIR)))
    print('--datadir {}'.format(os.path.abspath(out_dir)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
