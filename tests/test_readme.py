"""Checks on README.md and THIRD_PARTY_NOTICES.md.

The README is the only document a new user reads, so these tests keep it true:
every section is there, every flag it prints is a flag the code accepts, every
file it names and every figure it shows exists, and it holds no private path
and no key.

The release documents hold no `TODO(` marker, and a test checks that.
"""
import contextlib
import io
import os
import re
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 has no tomllib
    tomllib = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO_ROOT, 'README.md')
JUPYTER_README = os.path.join(REPO_ROOT, 'Jupyter', 'README.md')
NOTICES = os.path.join(REPO_ROOT, 'THIRD_PARTY_NOTICES.md')
PYPROJECT = os.path.join(REPO_ROOT, 'pyproject.toml')

# The two documents a new user reads. README.md is the front door; the
# notebook walkthrough lives in Jupyter/README.md so the top-level README
# stays short.
DOCS = (README, JUPYTER_README)
DOC_IDS = ['README.md', 'Jupyter/README.md']

# The headings the release README must keep, in the order it uses them.
REQUIRED_SECTIONS = [
    '## Abstract',
    '## What is C-SONIC',
    '## Results',
    '## Installation',
    '## Pretrained models',
    '## Data',
    '## Demo',
    '## Training',
    '## Usage in your own code',
    '## Tests',
    '## Reproducibility notes',
    '## Citation',
    '## License and third-party code',
]

# The command line scripts the README shows. Every flag it gives one of them
# must come from that script's own parser.
SCRIPTS = ['train.py', 'CSONIC_test.py', 'detection.py', 'make_pairs.py',
           'extract_sample.py']

# The files the path check applies to. A dataset file or a checkpoint is not in
# the repository, so only the documents and the source files are checked.
SOURCE_SUFFIXES = ('.py', '.ipynb')
SOURCE_NAMES = ('pyproject.toml', 'LICENSE', 'THIRD_PARTY_NOTICES.md')

# Directories the README names that a new clone does not hold. The user
# downloads each one, or a run makes it.
CREATED_DIRS = ('pretrained/', 'figs/', 'logs/', 'checkpoints/', 'vis/',
                'samples/')

# The kinds of TODO marker the documents may still hold. The release is
# complete, so the set is empty: any marker is an error.
DOCUMENTED_TODOS = frozenset()

# The largest figure the README may show. A bigger one makes the clone slow.
MAX_FIGURE_BYTES = 500000

# A private path, or a key written into the document.
FORBIDDEN_STRINGS = ('/home/', '/media/', '/Users/', 'C:\\Users',
                     'wandb.login(key=')
SECRET_ASSIGNMENT = re.compile(
    r'(?i)([\w.\-]*(?:key|token|secret|password))\s*=\s*([^\s`\'")]+)')
# The variables the README documents by name. A reader must set these.
ALLOWED_VARIABLES = ('WANDB_API_KEY',)
ALLOWED_VARIABLE_PREFIXES = ('CSONIC_',)

_FLAG_CACHE = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def read_text(path):
    """Return the contents of one file."""
    with open(path, encoding='utf-8') as handle:
        return handle.read()


def read_readme():
    """Return the README text."""
    return read_text(README)


def strip_code_fences(text):
    """Return the text without the fenced code blocks."""
    return re.sub(r'^```.*?^```', '', text, flags=re.S | re.M)


def fenced_lines(text):
    """Return the lines inside the fenced code blocks."""
    lines = []
    for block in re.findall(r'^```.*?^```', text, flags=re.S | re.M):
        lines.extend(block.splitlines()[1:-1])
    return lines


def inline_spans(text):
    """Return the contents of every single backtick span, fences removed."""
    return re.findall(r'`([^`\n]+)`', strip_code_fences(text))


def flags_in(text):
    """Return every long flag written in the text."""
    return set(re.findall(r'--[A-Za-z][\w-]*', text))


def _help_text(parse_function):
    """Return what a parse function prints for --help, without leaving pytest."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        with contextlib.suppress(SystemExit):
            parse_function(['--help'])
    return out.getvalue() + err.getvalue()


def _help_text_from_subprocess(script):
    """Return what the script prints for --help. The parser stays the source."""
    result = subprocess.run([sys.executable, script, '--help'], cwd=REPO_ROOT,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, \
        '{} --help failed with {}:\n{}'.format(script, result.returncode,
                                               result.stderr[-2000:])
    return result.stdout + result.stderr


def script_flags(script):
    """Return the long flags one repository script accepts.

    The flags come from the live parser, never from a list written here, so a
    renamed flag makes the README test fail instead of the user's first run.
    """
    if script in _FLAG_CACHE:
        return _FLAG_CACHE[script]

    text = ''
    if script == 'train.py':
        import config
        text = _help_text(config.get_args)
    elif script == 'CSONIC_test.py':
        import CSONIC_test
        for name in ('build_parser', 'get_parser'):
            builder = getattr(CSONIC_test, name, None)
            if callable(builder):
                text = builder().format_help()
                break
        else:
            parse_function = getattr(CSONIC_test, 'parse_args', None)
            if callable(parse_function):
                text = _help_text(parse_function)
    elif script == 'make_pairs.py':
        import make_pairs
        text = make_pairs.build_parser().format_help()
    elif script == 'extract_sample.py':
        import extract_sample
        text = extract_sample.build_parser().format_help()

    if not flags_in(text):
        text = _help_text_from_subprocess(script)

    flags = flags_in(text)
    assert flags, 'no flags found in the --help output of {}'.format(script)
    _FLAG_CACHE[script] = flags
    return flags


def readme_commands(text, script):
    """Return the README command lines that call one repository script.

    A backslash at the end of a line joins it with the next one first, so a
    wrapped command stays one command.
    """
    joined = re.sub(r'\\\s*\n\s*', ' ', text)
    pattern = re.compile(r'(^|[\s/])' + re.escape(script) + r'(\s|$)')
    return [line for line in joined.splitlines() if pattern.search(line)]


def clean_token(token):
    """Return one whitespace-separated token without its decoration.

    A quote and a punctuation mark can sit on the same edge, one behind the
    other (for example a comma right outside a closing quote in
    ``'path.npy',``), so the two strips alternate until neither changes the
    token.
    """
    token = token.lstrip('│├└─┬┐┌┘ \t')
    previous = None
    while token != previous:
        previous = token
        token = token.strip('`"\'')
        token = token.rstrip('.,;:()[]{}')
    return token


def path_tokens(text):
    """Return the path-like tokens of the README, fenced blocks included."""
    fragments = inline_spans(text) + fenced_lines(text)
    tokens = set()
    for fragment in fragments:
        for token in fragment.split():
            token = clean_token(token)
            if token and ('<' not in token and '>' not in token):
                tokens.add(token)
    return tokens


def is_source_path(token):
    """True when the token names a document or a source file of the repository."""
    return token.endswith(SOURCE_SUFFIXES) or token in SOURCE_NAMES


def is_created_dir(token):
    """True when the token sits in a folder the user downloads or a run makes."""
    first = token.split('/')[0] + '/'
    return first in CREATED_DIRS


def inside_repo(token):
    """True when the token resolves to a path inside the repository."""
    resolved = os.path.realpath(os.path.join(REPO_ROOT, token))
    root = os.path.realpath(REPO_ROOT)
    return resolved == root or resolved.startswith(root + os.sep)


def todo_labels(text):
    """Return the label of every TODO marker in the text.

    The label is what stands before the first colon, so
    ``TODO(license: confirm ...)`` gives ``license``.
    """
    labels = []
    for body in re.findall(r'TODO\(([^)]*)\)', text):
        labels.append(body.split(':')[0].strip())
    return labels


def pyproject_value(key):
    """Return one string value of the [project] table of pyproject.toml."""
    text = read_text(PYPROJECT)
    if tomllib is not None:
        return tomllib.loads(text)['project'][key]

    # Python 3.10 has no tomllib, so read the one value this test needs.
    table = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            table = line[1:-1]
        elif table == 'project':
            match = re.match(re.escape(key) + r'\s*=\s*["\']([^"\']+)["\']', line)
            if match:
                return match.group(1)
    raise AssertionError('no {} in the [project] table of pyproject.toml'.format(key))


def pyproject_version():
    """Return the version in the [project] table of pyproject.toml."""
    return pyproject_value('version')


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def test_readme_has_required_sections():
    """Every heading of the release README is present, and in order."""
    lines = [line.strip() for line in read_readme().splitlines()]

    missing = [section for section in REQUIRED_SECTIONS if section not in lines]
    assert missing == [], 'missing README sections: {}'.format(missing)

    positions = [lines.index(section) for section in REQUIRED_SECTIONS]
    assert positions == sorted(positions), 'the README sections are out of order'


def test_readme_banner_points_at_a_figure():
    """The banner shows a figure, or it carries the marker that asks for one."""
    text = read_readme()
    first = text.splitlines()[0].strip()

    match = re.match(r'^!\[[^\]]*\]\((figs/\S+)\)$', first)
    assert match, 'the first line must be the banner image, got: {}'.format(first)

    target = match.group(1)
    if not os.path.exists(os.path.join(REPO_ROOT, target)):
        assert 'TODO(add figure)' in text, \
            '{} does not exist, so the README must keep TODO(add figure)'.format(target)


def test_readme_figures_exist():
    """Every figure the README shows is in the repository, and it is small.

    A figure that is not there gives the reader a broken image. A large figure
    makes the clone slow, so each one stays at or below 500000 bytes.
    """
    text = read_readme()
    assert not re.search(r'!\[[^\]]*\]\[', text), \
        'use inline image links, not reference-style ones, so this test sees them'
    targets = re.findall(r'!\[[^\]]*\]\(([^)\s]+)\)', text)
    assert targets, 'the README shows no figure'

    outside = [name for name in targets if not name.startswith('figs/')]
    assert outside == [], 'every figure must be a file in figs/, got: {}'.format(outside)

    missing = [name for name in targets
               if not os.path.isfile(os.path.join(REPO_ROOT, name))]
    assert missing == [], 'the README shows figures that do not exist: {}'.format(missing)

    too_big = ['{} ({} bytes)'.format(name, os.path.getsize(
        os.path.join(REPO_ROOT, name))) for name in targets
        if os.path.getsize(os.path.join(REPO_ROOT, name)) > MAX_FIGURE_BYTES]
    assert too_big == [], 'these figures are larger than {} bytes: {}'.format(
        MAX_FIGURE_BYTES, too_big)


def test_readme_front_matter_is_filled():
    """The title, the author, the collaborators, the paper link and the abstract.

    A TODO marker passes, and so does the real text that replaces it. Only an
    empty field fails.
    """
    lines = [line.rstrip() for line in read_readme().splitlines()]

    titles = [line for line in lines if re.match(r'^#\s+\S', line)]
    assert titles, 'the README has no title'
    assert '## Abstract' in lines, 'the README has no abstract'
    start = lines.index('## Abstract')
    front_lines = lines[lines.index(titles[0]) + 1:start]

    # The thesis line names its one author ("... by Akshay A. Hinduja, ...").
    # A separate author heading would name the author a second time.
    front = ' '.join(front_lines)
    assert re.search(r'\bby [A-Z][\w.]*(?: [A-Z][\w.]*)+', front), \
        'the front matter names no author ("by <Name>")'
    headings = [line for line in front_lines if re.match(r'^###\s+\S', line)]
    assert headings == [], 'the author is named twice: {}'.format(headings)
    assert any(re.match(r'^Collaborators:\s*\S', line) for line in front_lines), \
        'the front matter lists no collaborators'

    links = [line for line in lines if re.match(r'^paperurl:\s*\S', line)]
    assert links, 'the README has no paperurl line'

    body = [line.strip() for line in lines[start + 1:] if line.strip()]
    assert body, 'the README has no abstract'
    assert not body[0].startswith('#'), 'the abstract section is empty'


def test_readme_todo_markers_are_documented():
    """The documents hold no TODO marker of a kind that DOCUMENTED_TODOS lacks."""
    labels = set()
    for path in (README, NOTICES, JUPYTER_README):
        labels |= set(todo_labels(read_text(path)))

    unknown = sorted(labels - DOCUMENTED_TODOS)
    assert unknown == [], 'undocumented TODO markers: {}'.format(unknown)

    if not DOCUMENTED_TODOS:
        bare = [os.path.relpath(path, REPO_ROOT)
                for path in (README, NOTICES, JUPYTER_README)
                if re.search(r'\bTODO\b', read_text(path))]
        assert bare == [], 'these documents still say TODO: {}'.format(bare)


def _existing_docs():
    """Return the text of every document of DOCS that exists on disk."""
    return [read_text(path) for path in DOCS if os.path.isfile(path)]


@pytest.mark.parametrize('script', SCRIPTS)
def test_readme_cli_flags_exist(script):
    """Every flag a doc gives a script is a flag that script accepts.

    A script's example commands can live in README.md or in
    Jupyter/README.md (the notebook walkthrough moved extract_sample.py
    there); the check looks at the union of both documents.
    """
    accepted = script_flags(script)

    commands = []
    for text in _existing_docs():
        commands += readme_commands(text, script)
    assert commands, 'no document shows a command for {}'.format(script)

    used = set()
    for line in commands:
        used |= flags_in(line)
    assert used, 'the docs give {} no flag'.format(script)

    unknown = sorted(flag for flag in used if flag not in accepted)
    assert unknown == [], '{} does not accept: {}'.format(script, unknown)


@pytest.mark.parametrize('path', DOCS, ids=DOC_IDS)
def test_readme_inline_flags_exist(path):
    """A flag named on its own in backticks is accepted by one of the scripts."""
    quoted = {span.split()[0] for span in inline_spans(read_text(path))
              if span.startswith('--')}
    assert quoted, '{} names no flag in backticks'.format(
        os.path.relpath(path, REPO_ROOT))

    accepted = set()
    for script in SCRIPTS:
        accepted |= script_flags(script)

    unknown = sorted(flag for flag in quoted if flag not in accepted)
    assert unknown == [], 'no script accepts: {}'.format(unknown)


@pytest.mark.parametrize('path', DOCS, ids=DOC_IDS)
def test_readme_paths_exist(path):
    """Every file and folder of the repository that a doc names is there.

    The check reads the backtick spans and the fenced blocks. A folder the user
    downloads, or a run makes, is not in a new clone, so only its position is
    checked. Paths always resolve against REPO_ROOT, even inside
    Jupyter/README.md, so that document writes its paths repo-root-relative
    too (for example `Jupyter/functions.py`, not `functions.py`).
    """
    tokens = path_tokens(read_text(path))

    files = sorted(token for token in tokens if is_source_path(token))
    assert files, '{} names no source file'.format(os.path.relpath(path, REPO_ROOT))

    missing = [name for name in files
               if not os.path.exists(os.path.join(REPO_ROOT, name))]
    assert missing == [], '{} names files that do not exist: {}'.format(
        os.path.relpath(path, REPO_ROOT), missing)

    folders = sorted(token for token in tokens
                     if token.endswith('/') and not is_created_dir(token))
    missing_folders = [name for name in folders
                       if not os.path.isdir(os.path.join(REPO_ROOT, name))]
    assert missing_folders == [], \
        '{} names folders that do not exist: {}'.format(
        os.path.relpath(path, REPO_ROOT), missing_folders)

    outside = sorted(token for token in files + folders if not inside_repo(token))
    assert outside == [], '{} names paths outside the repository: {}'.format(
        os.path.relpath(path, REPO_ROOT), outside)


def test_readme_links_the_third_party_notices():
    """The license section sends the reader to the notices, which exist."""
    text = read_readme()

    assert os.path.isfile(NOTICES), 'THIRD_PARTY_NOTICES.md is missing'
    assert 'THIRD_PARTY_NOTICES.md' in text, \
        'the README must link THIRD_PARTY_NOTICES.md'

    notices = read_text(NOTICES)
    for name in ('SONIC', 'CAPS', 'SuperPoint', 'torchvision'):
        assert name in notices, 'THIRD_PARTY_NOTICES.md does not name {}'.format(name)


def test_readme_has_no_user_paths_or_secrets():
    """The documents hold no private path and no key."""
    for path in (README, NOTICES, JUPYTER_README):
        text = read_text(path)
        for forbidden in FORBIDDEN_STRINGS:
            assert forbidden not in text, \
                '{} holds {!r}'.format(os.path.basename(path), forbidden)

        found = []
        for name, value in SECRET_ASSIGNMENT.findall(text):
            upper = name.upper()
            if upper in ALLOWED_VARIABLES:
                continue
            if upper.startswith(ALLOWED_VARIABLE_PREFIXES):
                continue
            found.append('{}={}'.format(name, value))
        assert found == [], \
            '{} assigns a key or a token: {}'.format(os.path.basename(path), found)


def test_the_project_is_named_c_sonic():
    """The project, the lock file and every hint use the repository name c-sonic.

    The GitHub repository is rpl-cmu/c-sonic. It was first created as
    cross-sonic. No file that a user reads or runs may still carry that name.
    """
    assert pyproject_value('name') == 'c-sonic', \
        'pyproject.toml names the project {}'.format(pyproject_value('name'))

    stale = []
    for name in ('README.md', 'pyproject.toml', 'uv.lock', 'config.py',
                 os.path.join('Jupyter', 'README.md'), 'THIRD_PARTY_NOTICES.md'):
        if 'cross-sonic' in read_text(os.path.join(REPO_ROOT, name)).lower():
            stale.append(name)
    # Only the cells of the notebook: its metadata holds the name of the local
    # kernel, which comes from the virtual environment of the user.
    import json
    notebook = json.loads(read_text(os.path.join(REPO_ROOT, 'Jupyter', 'Visualizer.ipynb')))
    cells = ''.join(''.join(cell.get('source', [])) for cell in notebook['cells'])
    if 'cross-sonic' in cells.lower():
        stale.append('Jupyter/Visualizer.ipynb')
    assert stale == [], 'these files still say cross-sonic: {}'.format(stale)


def test_version_matches_pyproject():
    """The package version and the project version agree."""
    import CSONIC

    assert CSONIC.__version__ == pyproject_version()


def test_docs_use_the_dataset_layout():
    """The docs point at the released sample, not the real-world tank layout.

    H0/, L0/ and YOUR_DIR_HERE are examples from the real-world tank dataset
    used during development, not from the simulated dataset this release
    ships. A public reader only has the released sample
    (samples/csonic_sample/), so every example must use that layout instead.
    """
    for path in DOCS:
        if not os.path.isfile(path):
            continue
        text = read_text(path)
        found = [token for token in ('YOUR_DIR_HERE', 'H0/', 'L0/') if token in text]
        assert found == [], '{} still shows the tank layout: {}'.format(
            os.path.relpath(path, REPO_ROOT), found)


def test_docs_sample_paths_exist():
    """Every samples/csonic_sample file a doc names is really there.

    The released sample is gitignored, so this test only runs when it is
    present on the machine (it is on this one).
    """
    sample_root = os.path.join(REPO_ROOT, 'samples', 'csonic_sample')
    if not os.path.isdir(sample_root):
        pytest.skip('samples/csonic_sample is not on this machine')

    checked = []
    for path in DOCS:
        if not os.path.isfile(path):
            continue
        for token in path_tokens(read_text(path)):
            if (token.startswith('samples/csonic_sample/')
                    and os.path.splitext(token)[1]):
                checked.append(token)

    assert checked, 'no document names a samples/csonic_sample file'

    missing = sorted(token for token in checked
                     if not os.path.isfile(os.path.join(REPO_ROOT, token)))
    assert missing == [], 'these sample paths do not exist: {}'.format(missing)


def heading_slug(heading):
    """Return the GitHub slug of one heading's text.

    Lower-case, drop everything that is not a letter, a digit, a space, a
    hyphen or an underscore, then turn every space into a hyphen.
    """
    slug = heading.strip().lower()
    slug = re.sub(r'[^a-z0-9 _-]', '', slug)
    return slug.replace(' ', '-')


def heading_slugs(text):
    """Return the slug of every heading of the text."""
    return [heading_slug(match.group(1)) for match in
            re.finditer(r'^#{1,6}\s+(.+)$', text, flags=re.M)]


def test_docs_links_resolve():
    """Every in-page link matches a real heading, every relative link exists.

    ``](#anchor)`` must match the GitHub slug of a heading in the same
    document. ``](path)``, with path neither an http(s) URL nor an anchor,
    must resolve to a real file relative to the document's own folder.
    """
    for path in DOCS:
        if not os.path.isfile(path):
            continue
        text = read_text(path)
        slugs = set(heading_slugs(text))
        folder = os.path.dirname(path)
        name = os.path.relpath(path, REPO_ROOT)

        for target in re.findall(r'\]\(([^)]+)\)', strip_code_fences(text)):
            target = target.strip()
            if target.startswith('#'):
                anchor = target[1:]
                assert anchor in slugs, \
                    '{} links to #{}, which matches no heading'.format(name, anchor)
            elif target.startswith(('http://', 'https://')):
                continue
            else:
                resolved = os.path.normpath(os.path.join(folder, target))
                assert os.path.exists(resolved), \
                    '{} links to {}, which does not exist'.format(name, target)
