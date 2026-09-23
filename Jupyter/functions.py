"""Interactive correspondence viewer for the C-SONIC model.

The viewer draws a sonar image pair side by side. Click a point in the left
image; the model predicts where that point lands in the right image and draws it
there, with a circle for the spread of the prediction.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# The notebook runs from the Jupyter directory, so put the repository root on
# the path before the project modules are imported.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CSONIC.csonic_model import CSONICModel  # noqa: E402
from dataloader.sonardata import SonarDataLoader, denormalize_image  # noqa: E402

# Width in pixels of the white strip between the two images.
GAP = 5


class Visualization(object):
    """Show a sonar image pair and the correspondences the model predicts."""

    def __init__(self, args):
        self.dataloader = SonarDataLoader(args).load_data()
        self.model = CSONICModel(args)
        self.loader_iter = iter(self.dataloader)
        self.sample = None
        self.fig = None
        self.ax = None
        self.cid = None
        self.coords = []
        self.colors = []
        self.with_std = False

    def random_sample(self):
        """Take the next pair from the loader. The loader shuffles the pairs."""
        self.sample = next(self.loader_iter)

    def plot_img_pair(self, with_std=False):
        """Draw the image pair side by side and listen for clicks.

        Set with_std to draw a circle around each predicted point. The radius
        grows with the spread the model reports.
        """
        self.coords = []
        self.colors = []
        self.with_std = with_std

        # The loader normalizes the images for the network. Undo that, so the
        # figure shows the pixels as they were recorded.
        im1 = denormalize_image(self.sample['im1'][0, 0]).cpu().numpy()
        im2 = denormalize_image(self.sample['im2'][0, 0]).cpu().numpy()
        self.h, self.w = im1.shape
        blank = np.ones((self.h, GAP))
        out = np.concatenate((im1, blank, im2), 1)

        # Open a new figure for every pair, and close the old one. With the
        # widget backend a cell that runs again clears its output, which removes
        # the view of the old figure; a draw into it shows nothing. The new
        # figure appears in the output of the cell that made it.
        if self.fig is not None:
            if self.cid is not None:
                self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(12, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(out, cmap='gray')
        self.ax.axis('off')
        self.fig.tight_layout()

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        """Mark the clicked query point, then find and draw its match.

        The query point must lie inside image 1, the left one. Every other
        click, including one in the gap or in image 2, is rejected.
        """
        x, y = event.xdata, event.ydata
        if (event.inaxes is not self.ax or x is None or y is None
                or not (0 <= x < self.w) or not (0 <= y < self.h)):
            print('Click inside image 1, the left one. The query point must lie there.')
            return

        coord = [float(x), float(y)]
        color = tuple(np.random.rand(3).tolist())
        self.coord = coord
        self.color = color
        self.coords.append(coord)
        self.colors.append(color)
        self.ax.scatter(coord[0], coord[1], color=color)
        self.find_correspondence()
        self.plot_correspondence()

    def find_correspondence(self):
        """Ask the model where the clicked point lands in image 2."""
        # Copy the batch, so the query points the loader produced stay intact.
        data_in = dict(self.sample)
        data_in['coord1'] = torch.tensor([[self.coord]], dtype=torch.float32,
                                         device=self.model.device)
        self.model.set_input(data_in)
        coord2_e, std = self.model.test()
        # Pixel coordinates in image 2, and the spread of the prediction.
        self.correspondence = coord2_e.squeeze().cpu().numpy()
        self.std = std.squeeze().cpu().numpy()

    def plot_correspondence(self):
        """Draw the predicted point in the right half of the figure."""
        x = self.correspondence[0] + self.w + GAP
        y = self.correspondence[1]
        self.ax.scatter(x, y, color=self.color)
        if self.with_std:
            circle = plt.Circle((x, y), radius=100 * self.std, fill=False,
                                color=self.color)
            self.ax.add_patch(circle)
        self.fig.canvas.draw_idle()
