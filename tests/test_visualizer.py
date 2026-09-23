"""Tests for the interactive visualizer in Jupyter/functions.py.

The visualizer is a notebook tool, so these tests drive it headless: they force
the Agg backend and call the click handler with a fake mouse event.
"""
import json
import os
import types

import matplotlib

matplotlib.use('Agg')  # the visualizer must work without a display

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import config
import Jupyter.functions as functions
from Jupyter.functions import GAP, Visualization

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(REPO_ROOT, 'Jupyter', 'Visualizer.ipynb')
CKPT = os.path.join(REPO_ROOT, 'pretrained', 'CSONIC_pretrained.pth')
SUPERPOINT_WEIGHTS = os.path.join(REPO_ROOT, 'pretrained', 'superpoint_v1.pth')

# The click test always writes its figure into the pytest temporary directory.
# Set CSONIC_ARTIFACT_DIR to also keep a copy somewhere you can look at it.
ARTIFACT_DIR = os.environ.get('CSONIC_ARTIFACT_DIR')


def _code_cells():
    """Return the code cells of the visualizer notebook."""
    with open(NOTEBOOK) as handle:
        notebook = json.load(handle)
    return [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']


# --------------------------------------------------------------------------
# the notebook
# --------------------------------------------------------------------------

def test_notebook_is_clean_and_points_at_csonic():
    """The notebook ships without outputs and drives the C-SONIC model."""
    cells = _code_cells()
    assert cells, 'the notebook has no code cells'
    for cell in cells:
        assert cell.get('outputs') == []
        assert cell.get('execution_count') is None

    source = '\n'.join(''.join(cell['source']) for cell in cells)
    assert 'CSONIC.csonic_model' in source
    assert 'CSONIC_pretrained.pth' in source
    assert 'csonic_sample' in source, 'the notebook must point at the demo sample'
    for stale in ('SONIC.sonic_model', 'CAPS', '/home/akshay'):
        assert stale not in source, 'the notebook still mentions {}'.format(stale)

    # The first cell walks up from the working directory to find the checkout,
    # and says so when it cannot.
    assert 'parents' in source
    assert ('Start Jupyter inside the c-sonic checkout (or %cd into it) '
            'so the repository root can be found') in source


# --------------------------------------------------------------------------
# clicks, against a fake loader and a fake model
# --------------------------------------------------------------------------

def _fake_batch(h=64, w=64):
    """One batch with the keys the model reads."""
    batch = {
        'im1': torch.zeros(1, 1, h, w),
        'im2': torch.ones(1, 1, h, w),
        'coord1': torch.zeros(1, 4, 2),
        'T1': torch.eye(4).unsqueeze(0),
        'T2': torch.eye(4).unsqueeze(0),
    }
    for prefix in ('im1', 'im2'):
        batch.update({
            prefix + '_width': torch.tensor([float(w)]),
            prefix + '_height': torch.tensor([float(h)]),
            prefix + '_r_min': torch.tensor([0.1]),
            prefix + '_r_max': torch.tensor([7.0]),
            prefix + '_elev': torch.tensor([12.0]),
            prefix + '_azi': torch.tensor([60.0]),
        })
    return batch


class _FakeLoader:
    """Stands in for SonarDataLoader. Serves two identical batches."""

    def __init__(self, args):
        self.args = args

    def load_data(self):
        return [_fake_batch(), _fake_batch()]


class _FakeModel:
    """Stands in for CSONICModel. Records every batch it is given."""

    def __init__(self, args):
        self.args = args
        self.device = 'cpu'
        self.inputs = []

    def set_input(self, data):
        self.inputs.append(data)

    def test(self):
        return torch.tensor([[[12.0, 34.0]]]), torch.tensor([[0.25]])


def _event(x, y, ax):
    return types.SimpleNamespace(xdata=x, ydata=y, inaxes=ax)


def test_replot_makes_a_new_figure(monkeypatch):
    """A second plot opens a new figure and closes the old one.

    With the widget backend a cell that runs again clears its output, which
    removes the view of the old figure. A draw into that figure shows nothing.
    A new figure appears in the output of the cell that made it.
    """
    monkeypatch.setattr(functions, 'SonarDataLoader', _FakeLoader)
    monkeypatch.setattr(functions, 'CSONICModel', _FakeModel)

    viz = Visualization(args=None)
    viz.random_sample()
    viz.plot_img_pair()
    first = viz.fig
    assert plt.fignum_exists(first.number)

    viz.random_sample()
    viz.plot_img_pair()
    assert viz.fig is not first, 'the second plot reused the old figure'
    # Matplotlib hands the freed number to the new figure, so the check looks
    # for the old figure object among the open figures.
    open_figures = [plt.figure(number) for number in plt.get_fignums()]
    assert first not in open_figures, 'the old figure stays open'
    assert viz.fig in open_figures
    assert viz.ax.figure is viz.fig
    # The click handler is registered on the new canvas: dispatch a real
    # mouse event through the canvas, not by a direct call of the handler.
    assert viz.cid is not None
    from matplotlib.backend_bases import MouseEvent
    x_pixel, y_pixel = viz.ax.transData.transform((10.0, 20.0))
    event = MouseEvent('button_press_event', viz.fig.canvas, x_pixel, y_pixel,
                       button=1)
    viz.fig.canvas.callbacks.process('button_press_event', event)
    assert len(viz.coords) == 1
    assert viz.coords[0] == pytest.approx([10.0, 20.0], abs=0.5)
    plt.close(viz.fig)


def test_click_handling_with_mocked_model(monkeypatch):
    """The click handler sends one query point, and rejects every bad click."""
    monkeypatch.setattr(functions, 'SonarDataLoader', _FakeLoader)
    monkeypatch.setattr(functions, 'CSONICModel', _FakeModel)

    viz = Visualization(args=None)
    viz.random_sample()
    viz.plot_img_pair(with_std=True)
    assert (viz.h, viz.w) == (64, 64)

    loaded_coords = viz.sample['coord1'].clone()
    viz.onclick(_event(10.0, 20.0, viz.ax))

    assert viz.coords == [[10.0, 20.0]]
    sent = viz.model.inputs[-1]['coord1']
    assert sent.shape == (1, 1, 2)
    assert sent.tolist() == [[[10.0, 20.0]]]
    # The query points the loader produced stay as they were.
    torch.testing.assert_close(viz.sample['coord1'], loaded_coords)

    assert viz.correspondence.tolist() == [12.0, 34.0]
    assert float(viz.std) == pytest.approx(0.25)
    drawn = viz.ax.collections[-1].get_offsets()[0]
    assert drawn[0] == pytest.approx(viz.correspondence[0] + GAP + viz.w)
    assert drawn[1] == pytest.approx(viz.correspondence[1])

    # None of these is a query point in image 1.
    other_ax = Figure().add_subplot(111)
    rejected = [
        _event(viz.w + 1.0, 10.0, viz.ax),        # the gap between the images
        _event(viz.w + 50.0, 10.0, viz.ax),       # image 2
        _event(-5.0, 10.0, viz.ax),               # left of image 1
        _event(10.0, -5.0, viz.ax),               # above image 1
        _event(10.0, viz.h + 1.0, viz.ax),        # below image 1
        _event(10.0, 10.0, other_ax),             # another axes
        _event(None, None, None),                 # outside the axes
    ]
    calls = len(viz.model.inputs)
    for event in rejected:
        viz.onclick(event)
    assert viz.coords == [[10.0, 20.0]]
    assert len(viz.model.inputs) == calls

    # A second sample opens a new figure and forgets the old points.
    first = viz.fig
    viz.random_sample()
    viz.plot_img_pair()
    assert viz.fig is not first
    assert viz.coords == []
    plt.close(viz.fig)


# --------------------------------------------------------------------------
# a real click
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_visualization_headless_click(tiny_dataset_dir, tmp_path, capsys):
    """A click in image 1 gets a correspondence in image 2 and draws it."""
    if not os.path.isfile(CKPT):
        pytest.skip('checkpoint not found at {}'.format(CKPT))

    args = config.get_args([])
    args.datadir = tiny_dataset_dir
    args.phase = 'test'
    args.batch_size = 1
    args.workers = 0
    args.ckpt_path = CKPT
    args.superpoint_weights = SUPERPOINT_WEIGHTS
    args.pretrained = 0

    viz = Visualization(args)
    viz.random_sample()
    viz.plot_img_pair(with_std=True)
    assert (viz.h, viz.w) == (512, 512)

    # A click on the left half is a query point in image 1.
    viz.onclick(_event(200.0, 300.0, viz.ax))
    assert viz.coords == [[200.0, 300.0]]

    correspondence = viz.correspondence
    assert correspondence.shape == (2,)
    assert np.all(np.isfinite(correspondence))
    # find_correspondence stores image 2 pixels; the shift into the concatenated
    # figure happens only when the point is drawn.
    assert 0 <= correspondence[0] < viz.w
    assert 0 <= correspondence[1] < viz.h

    std = float(viz.std)
    assert np.isfinite(std) and std >= 0

    # The drawn point sits in the right half of the figure.
    drawn = viz.ax.collections[-1].get_offsets()[0]
    assert drawn[0] == pytest.approx(correspondence[0] + viz.w + GAP)
    assert drawn[1] == pytest.approx(correspondence[1])

    # The gap, image 2 and a click outside the axes are all rejected.
    viz.onclick(_event(viz.w + 1.0, 100.0, viz.ax))
    viz.onclick(_event(viz.w + 50.0, 100.0, viz.ax))
    viz.onclick(_event(None, None, None))
    assert viz.coords == [[200.0, 300.0]]

    with capsys.disabled():
        print('\ncorrespondence (image 2 pixels): {}, std: {:.6f}'
              .format(correspondence, std))

    out = tmp_path / 'click.png'
    viz.fig.savefig(out, dpi=150)
    assert out.stat().st_size > 0
    if ARTIFACT_DIR:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        artifact = os.path.join(ARTIFACT_DIR, 'click.png')
        viz.fig.savefig(artifact, dpi=150)
        with capsys.disabled():
            print('click figure written to {}'.format(artifact))
