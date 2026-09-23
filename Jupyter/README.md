# The C-SONIC notebook

`Jupyter/Visualizer.ipynb` shows correspondences between one pair of sonar
images. Click a point in the left image; the model predicts where that point
lands in the right image, and draws it there inside a circle that shows the
spread of the prediction. The viewer class that does this lives in
`Jupyter/functions.py`.

## Layout

```
pretrained/CSONIC_pretrained.pth   # the released C-SONIC model
pretrained/superpoint_v1.pth       # the SuperPoint detector
samples/csonic_sample/logs/        # csonic_sample_v1 from the C-SONIC Drive folder
```

Put the two checkpoints and the sample where the tree above shows, at the
root of the repository. See [the main README](../README.md) for where to
download the checkpoints.

## Start the notebook

Install the notebook packages once:

```bash
uv sync --group notebook
```

Then, in JupyterLab:

```bash
uv run jupyter lab Jupyter/Visualizer.ipynb
```

Or, in VS Code: open `Jupyter/Visualizer.ipynb` and pick the `.venv` kernel of
this repository, then run the cells from the toolbar.

Run the first cell once. It builds the model and the data loader. Run the
second cell for a pair: it picks a random pair, draws the two images side by
side, and turns on clicking. Run the second cell again for another pair; each
run replaces the figure. The clicks need `ipympl`, which the notebook group of
`uv sync` installs; if a click does nothing, check that the first line of the
second cell, `%matplotlib widget`, ran without error.

## The sample

`csonic_sample_v1` holds two scenes:

* `ELC2`, sonar 0, frames 1000, 1005 and 1010: 16 pairs (BPH against BPL, and
  BPL against the long-range log `ELC2_BPL_1_LR`, at the same frame or 5 frames
  apart, plus two BPL-against-BPL pairs 5 frames apart).
* `ELC1`, sonar 0, frames 1200, 1204 and 1208: 9 pairs (BPH against BPL at the
  same frame or 4 frames apart, plus two BPL-against-BPL pairs).

In each BPH pair, the high-frequency (BPH) image comes first. Its view (60
degrees) is narrower than the BPL view (130 degrees), so most query points in
the BPH image are inside the BPL view. The notebook shows the first image on the left, so you
click in the high-frequency image.

In total: 25 pairs, 45 `.npy` data files, about 31 MB. The notebook sets
`args.phase = 'test'`, so it reads the three `pairs*_val.txt` lists of the
sample, not `pairs.txt`.

## Making your own sample

`extract_sample.py` builds a sample like the released one from the log zip
files, without unpacking them: it reads single members of each archive and
writes them in the dataset layout, with the six pair lists. These two
commands made the released sample:

```bash
uv run python extract_sample.py --zips /path/to/logs-zip/ELC2_BPH_1.zip \
    /path/to/logs-zip/ELC2_BPL_1.zip /path/to/logs-zip/ELC2_BPL_1_LR.zip \
    --out samples/csonic_sample --preview
uv run python extract_sample.py --zips /path/to/logs-zip/ELC1_BPH_1.zip \
    /path/to/logs-zip/ELC1_BPL_1.zip --scene ELC1 --frames 1200 1204 1208 \
    --max-gap 4 --out samples/csonic_sample --preview --append
```

`--append` adds the new pairs to the lists that `--out` already holds;
without it, the tool refuses a folder that already holds the lists. `--preview`
also writes a PNG of every image, under `<out>/preview`. `--scene` picks a
scene of `make_pairs.py`'s table (`ELC1`, `ELC2` or `RW`); `--sonar` picks the
sensor, `--frames` picks the frames, and `--max-gap` bounds how far apart two
frames of a pair may be.

To use your own data, set `args.datadir` in the first cell to the dataset root,
the folder that holds `logs/`. See [the main README](../README.md) for the
dataset layout.
