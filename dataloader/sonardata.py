import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import dataloader.data_utils as data_utils

# Mean and standard deviation of the sonar training images, measured once over
# the training set.
SONAR_MEAN = 0.052
SONAR_STD = 0.060

rand = np.random.RandomState(234)


def denormalize_image(t):
    """Undo the training normalization. Returns pixel values back in [0, 1]."""
    return (t * SONAR_STD + SONAR_MEAN).clamp(0, 1)


class SonarDataLoader():
    """Wraps SonarData in a torch DataLoader."""

    def __init__(self, args):
        self.args = args
        self.dataset = SonarData(args)
        # Only the training set is shuffled; validation keeps the file order.
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=args.batch_size,
            shuffle=args.phase == 'train', num_workers=args.workers,
            collate_fn=self.my_collate)

    def my_collate(self, batch):
        """Collate a batch, dropping the items the dataset could not read."""
        batch = list(filter(lambda b: b is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'SonarDataLoader'

    def __len__(self):
        return len(self.dataset)


class SonarData(Dataset):
    """Sonar image pairs with their poses and sensor metadata.

    Each phase reads three parallel lists: image pairs, pose pairs and metadata
    pairs. Line n of each list describes the same pair.
    """

    def __init__(self, args):
        self.args = args
        self.phase = args.phase
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(SONAR_MEAN,), std=(SONAR_STD,)),
        ])

        if args.phase == 'train':
            self.root = args.datadir
            self.pair_files = (args.pairs_file, args.pairs_pos_file,
                               args.pairs_meta_file)
        else:
            self.root = args.val_data_dir or args.datadir
            self.pair_files = (args.val_pairs_file, args.val_pairs_pos_file,
                               args.val_pairs_meta_file)
        if not self.root:
            raise ValueError('--datadir is required')

        self.imf1s, self.imf2s, self.posf1s, self.posf2s, self.metaf1s, self.metaf2s = \
            self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))

        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])
        self.posf1s = list(np.array(self.posf1s)[index])
        self.posf2s = list(np.array(self.posf2s)[index])
        self.metaf1s = list(np.array(self.metaf1s)[index])
        self.metaf2s = list(np.array(self.metaf2s)[index])

    def _resolve(self, name):
        """Return the pair list path. A relative name sits under the root."""
        return name if os.path.isabs(name) else os.path.join(self.root, name)

    @staticmethod
    def _read_pair_list(path):
        """Read one pair list. Returns a list of (first, second) names.

        Every line must hold exactly two paths. A blank or malformed line is an
        error, not something to skip: the three lists are matched by position,
        so skipping line n of one file would silently pair the wrong images.
        Empty lines at the end of the file are allowed, so a trailing newline is
        fine.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError('pair file not found: {}'.format(path))
        with open(path, 'r') as handle:
            lines = handle.read().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()

        pairs = []
        for lineno, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    '{}:{}: expected two whitespace-separated paths, got {!r}'
                    .format(path, lineno, line))
            pairs.append((parts[0], parts[1]))
        return pairs

    def read_pairs(self):
        """Read the image, pose and metadata lists into six path lists."""
        print('reading image pairs from {}...'.format(self.root))
        images, poses, metas = (self._read_pair_list(self._resolve(name))
                                for name in self.pair_files)

        if not (len(images) == len(poses) == len(metas)):
            raise ValueError(
                'the pair files disagree in length: {} image pairs, {} pose pairs, '
                '{} metadata pairs'.format(len(images), len(poses), len(metas)))

        def under_root(pairs, column):
            return [os.path.join(self.root, pair[column]) for pair in pairs]

        return (under_root(images, 0), under_root(images, 1),
                under_root(poses, 0), under_root(poses, 1),
                under_root(metas, 0), under_root(metas, 1))

    def read_meta(self, im_meta):
        '''return the meta information'''
        '''{'width': 512, 'height': 512, 'r_min': 0.1, 'r_max': 10, 'elev': 12, 'azi': 60}'''
        meta_pkl = np.load(im_meta, allow_pickle=True)
        return meta_pkl.item()

    def read_default_pose(self, im_meta):
        '''return the default pose'''
        default_pose = np.asarray(np.load(im_meta))
        return default_pose

    @staticmethod
    def get_extrinsics(default_pose):
        '''change tranlation vector to be homogeneous -rcw as per Foundations of CV
        and the rotation matrix to have the axis direcitons as rows
        '''
        extrinsic = np.eye(4, 4)
        extrinsic[:3, :3] = default_pose[:3, :3].T
        extrinsic[:3, 3] = -default_pose[:3, :3].T @ default_pose[:3, 3]
        return extrinsic

    @staticmethod
    def get_relative_pose(pose2, pose1):
        '''get the relative rotation and translation between two poses'''
        relPose = pose2 @ np.linalg.inv(pose1)
        return relPose

    def __getitem__(self, item):
        '''get item'''
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        im1_pos = self.posf1s[item]
        im2_pos = self.posf2s[item]
        im1_meta = self.metaf1s[item]
        im2_meta = self.metaf2s[item]

        im1 = np.load(imf1)
        im2 = np.load(imf2)
        # Flip the images. The sensor origin sits at the top centre of the stored
        # image; the model expects it at the bottom centre.
        im1 = np.flip(im1).copy()
        im2 = np.flip(im2).copy()

        # gets the extrinsics using the pose information
        extrinsic1 = self.get_extrinsics(self.read_default_pose(im1_pos))
        extrinsic2 = self.get_extrinsics(self.read_default_pose(im2_pos))

        meta1 = self.read_meta(im1_meta)
        meta2 = self.read_meta(im2_meta)

        # take the image size from the metadata, not from the array
        h = meta1['height']
        w = meta1['width']

        # get the relative pose
        relative = self.get_relative_pose(extrinsic2, extrinsic1)

        # generate candidate query points
        coord1 = data_utils.generate_query_kpts(
            im1, self.args.num_pts, self.args.akaze_superpoint_pts, h, w,
            superpoint_weights=self.args.superpoint_weights)

        if len(coord1) < self.args.num_pts:
            print("coord1 has points {} but needed {}".format(len(coord1), self.args.num_pts))

        coord1 = torch.from_numpy(coord1).float()
        pose = torch.from_numpy(relative).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        return {'im1': im1_tensor,
                'im2': im2_tensor,
                'pose': pose,
                'T1': extrinsic1,
                'T2': extrinsic2,
                'coord1': coord1,
                'im1_width': meta1['width'],
                'im1_height': meta1['height'],
                'im2_width': meta2['width'],
                'im2_height': meta2['height'],
                'im1_r_min': meta1['r_min'],
                'im1_r_max': meta1['r_max'],
                'im2_r_min': meta2['r_min'],
                'im2_r_max': meta2['r_max'],
                'im1_elev': meta1['elev'],
                'im1_azi': meta1['azi'],
                'im2_elev': meta2['elev'],
                'im2_azi': meta2['azi'],
                }

    def __len__(self):
        '''returns the length '''
        return len(self.imf1s)
