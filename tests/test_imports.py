"""Import smoke tests: the repository must import with and without wandb."""
import importlib
import os
import re
import sys

import pytest

MODULES = [
    "config", "utils", "CSONIC", "CSONIC.network", "CSONIC.criterion",
    "CSONIC.csonic_model", "dataloader.sonardata", "dataloader.data_utils",
    "dataloader.demo_superpoint", "train", "Jupyter.functions",
    "detection", "CSONIC_test",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_public_names_exist():
    from CSONIC.csonic_model import CSONICModel  # noqa: F401
    from CSONIC.network import CSONICNet  # noqa: F401
    from dataloader.sonardata import SonarData, SonarDataLoader  # noqa: F401
    from train import train_sonardata  # noqa: F401


def test_imports_without_wandb(monkeypatch):
    for key in [k for k in sys.modules if k == "wandb" or k.startswith("wandb.")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.setitem(sys.modules, "wandb", None)  # makes `import wandb` raise ImportError
    for name in ("CSONIC.csonic_model", "train"):
        monkeypatch.delitem(sys.modules, name, raising=False)
        importlib.import_module(name)
    from CSONIC.csonic_model import _wandb
    assert _wandb() is None


WANDB_KEY_PLACEHOLDER = "YOUR_WANDB_API_KEY"

# A wandb API key is 40 lowercase hex characters.
KEY_SHAPE = re.compile(r"\b[0-9a-f]{40}\b")
KEY_CONTEXT = re.compile(r"(?i)wandb|api[_ ]?key")
# A key-shaped string within this many lines of a wandb or API-key mention
# counts, so a call split over several lines is still caught.
KEY_WINDOW = 3
TEXT_SUFFIXES = (".py", ".md", ".toml", ".txt", ".ipynb", ".yaml", ".yml", ".cfg",
                 ".ini", ".json", ".sh", ".env", ".lock", ".rst")
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "samples",
             "pretrained", "checkpoints", "logs", "outputs", "wandb"}
LOGIN_CALL = re.compile(r"wandb\s*\.\s*login\s*\(")
LOGIN_KEY = re.compile(r"wandb\s*\.\s*login\s*\(\s*key\s*=\s*([^)]*)\)", re.S)


def test_train_holds_only_the_wandb_key_placeholder():
    """train.py offers a commented login line with a placeholder, never a key."""
    src = open("train.py").read()
    keys = LOGIN_KEY.findall(src)
    assert keys, "train.py has no wandb.login placeholder"
    assert all(value.strip("\"' \n#") == WANDB_KEY_PLACEHOLDER for value in keys), keys
    for line in src.splitlines():
        if LOGIN_CALL.search(line):
            assert line.lstrip().startswith("#"), "the login placeholder must stay commented"
    assert not KEY_SHAPE.search(src), "train.py holds a key-shaped string"
    assert "/home/akshay" not in src and "/media/akshay" not in src


def key_shaped_lines(lines):
    """Return the numbers of the lines with a key-shaped string near a wandb mention."""
    context = [number for number, line in enumerate(lines) if KEY_CONTEXT.search(line)]
    found = []
    for number, line in enumerate(lines):
        if KEY_SHAPE.search(line) and any(abs(number - c) <= KEY_WINDOW for c in context):
            found.append(number + 1)
    return found


def test_key_scan_catches_a_split_call():
    """The scan sees a key on the line after the login call."""
    fake = "0123456789abcdef" * 2 + "01234567"
    assert key_shaped_lines(["wandb.login (", '    key="{}")'.format(fake)]) == [2]
    assert key_shaped_lines(["# see commit {}".format(fake)]) == []


def test_no_wandb_key_in_the_repository():
    """No text file holds a key-shaped string near a wandb or API-key mention."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if not name.endswith(TEXT_SUFFIXES):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8", errors="ignore") as handle:
                lines = handle.read().splitlines()
            found += ["{}:{}".format(os.path.relpath(path, root), n)
                      for n in key_shaped_lines(lines)]
    assert found == [], "key-shaped strings next to wandb: {}".format(found)
