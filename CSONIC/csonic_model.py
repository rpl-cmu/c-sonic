''' Derived from the original codebase of CAPSNet'''

import copy
import os

import torch
from torch import optim
from CSONIC.criterion import CtoFCriterion
from CSONIC.network import CSONICNet
from dataloader.sonardata import denormalize_image
from utils import make_matching_figure

META_KEYS = ('width', 'height', 'r_min', 'r_max', 'elev', 'azi')


def _meta_from_batch(data, prefix, device):
    """Build the sonar metadata dict from a batch. Each value is (B, 1)."""
    return {key: data['{}_{}'.format(prefix, key)].reshape(-1, 1).to(device)
            for key in META_KEYS}


def _wandb():
    """Return the wandb module only when it is installed and a run is active."""
    try:
        import wandb
    except ImportError:
        return None
    return wandb if wandb.run is not None else None


def zero_subnormal_weights(net):
    """Zero every float in the model that is too small to be a normal float.

    Half of the released checkpoint is subnormal. On the pairs tested, zeroing
    them left the matches bit-identical, but a CPU convolution that meets one
    drops off its fast path, and the forward pass goes from about a second to
    about a minute. :meth:`CSONICModel.load_model` calls this after every
    checkpoint it loads.
    Rewriting the weights removes the cause once, on whichever device the model
    sits, and changes nothing outside this model. Each tensor is tested
    against the smallest normal value of its own dtype. The integer buffers are
    left alone.

    :param net: the module to rewrite, in place
    :return: how many entries were zeroed
    """
    zeroed = 0
    with torch.no_grad():
        for tensor in list(net.parameters()) + list(net.buffers()):
            if not tensor.is_floating_point():
                continue
            tiny = torch.finfo(tensor.dtype).tiny
            subnormal = (tensor.abs() > 0) & (tensor.abs() < tiny)
            count = int(subnormal.sum())
            if count:
                tensor[subnormal] = 0
                zeroed += count
    return zeroed


class CSONICModel():

    def name(self):
        return 'C-SONIC Model'

    def __init__(self, args, device=None, zero_subnormal_weights=True):
        """Build the model, its optimizer and its loss.

        :param args: the parsed arguments
        :param device: where to build everything, for example ``'cpu'`` or
            ``'cuda:1'``. The default is the GPU when there is one. Passing it
            here, rather than moving the model afterwards, keeps the network,
            the optimizer state and the criterion on one device and allocates
            nothing on any other.
        :param zero_subnormal_weights: after a checkpoint loads, zero every
            weight smaller than the smallest normal float, which is what keeps
            CPU inference fast. See :func:`zero_subnormal_weights`. The count
            is kept in ``zeroed_subnormal``. False keeps the stored values.
        """
        self.args = args
        self.device = str(device) if device is not None else \
            ('cuda' if torch.cuda.is_available() else 'cpu')
        self.zero_subnormal_on_load = bool(zero_subnormal_weights)
        # How many subnormal entries the last checkpoint load zeroed.
        self.zeroed_subnormal = 0

        # Resolve the checkpoint before the network exists. A checkpoint supplies
        # every weight, so downloading the ImageNet initialisation first would
        # only overwrite it. The caller's namespace keeps its own pretrained
        # value; only the copy the network reads is changed.
        self.ckpt_to_load = self._resolve_checkpoint()
        net_args = args
        if self.ckpt_to_load is not None:
            net_args = copy.copy(args)
            net_args.pretrained = 0

        # init model, optimizer, scheduler
        self.model = CSONICNet(net_args, self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        # reloading from checkpoints
        self.start_step = self.load_from_ckpt()

        # define loss function
        self.criterion = CtoFCriterion(args).to(self.device)

    def set_input(self, data):
        """Move one batch onto the device and split out the metadata."""
        self.im1 = data['im1'].to(self.device)
        self.im2 = data['im2'].to(self.device)

        # Coordinates for image 1
        self.coord1 = data['coord1'].to(self.device)

        # Sonar extrinsics for the two images
        self.T1 = data['T1'].to(self.device)
        self.T2 = data['T2'].to(self.device)

        # Sensor geometry for the two images
        self.meta1 = _meta_from_batch(data, 'im1', self.device)
        self.meta2 = _meta_from_batch(data, 'im2', self.device)

    def forward(self):
        self.out = self.model(self.im1, self.im2, self.coord1)

    def val_forward(self):
        self.out = self.model(self.im1, self.im2, self.coord1)
        loss = self.criterion(self.coord1, self.out, self.T1, self.T2, self.im2, self.meta1, self.meta2)
        self.j_loss, self.eloss_c, self.eloss_f, self.closs_c, self.closs_f = loss
        return loss

    def backward_net(self):
        """Compute the loss and back-propagate it.

        j_loss is the total loss; the other four are the epipolar and cycle
        consistency terms at the coarse and the fine level.
        """
        loss = self.criterion(self.coord1, self.out, self.T1, self.T2, self.im2, self.meta1, self.meta2)
        self.j_loss, self.eloss_c, self.eloss_f, self.closs_c, self.closs_f = loss
        self.j_loss.backward()

    def optimize_parameters(self):
        """Run one training step."""
        self.optimizer.zero_grad()
        self.forward()
        self.backward_net()
        self.optimizer.step()
        self.scheduler.step()

    def test(self):
        '''This function returns the coordinates of the descriptors given the
        2 images and the coordinates for the first image
        '''
        self.model.eval()
        with torch.no_grad():
            coord2_e, std = self.model.test(self.im1, self.im2, self.coord1)
        return coord2_e, std
    
    def extract_features(self, im, coord, im_ref=None):
        '''
        Returns the coarse and fine feature descriptors for the image given the
        coordinates. Cross-attention needs the other image of the pair as im_ref.
        '''
        self.model.eval()
        with torch.no_grad():
            feat_c, feat_f = self.model.extract_features(im, coord, im_ref)
        return feat_c, feat_f
    
    def _log_images(self, n_iter, wb, prefix, remove):
        """Save the query points and their predicted matches, and log the figure."""
        num_kpts_display = min(self.args.num_kpts_shown, self.coord1[0].shape[0])
        # The images were normalized for the network; show them as pixels again.
        im1_o = denormalize_image(self.im1[0, 0]).cpu().numpy()
        im2_o = denormalize_image(self.im2[0, 0]).cpu().numpy()
        kpt1 = self.coord1.cpu().numpy()[0][:num_kpts_display, :]
        kpt2 = self.out['coord2_ef'].detach().cpu().numpy()[0][:num_kpts_display, :]

        vis_dir = os.path.join(self.args.outdir, self.args.exp_name, 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        local_path = os.path.join(vis_dir, f'debug_{prefix}{n_iter}.png')
        make_matching_figure(im1_o, im2_o, kpt1, kpt2, path=local_path)
        wb.log({f'{prefix}correspondence_image_{n_iter}': wb.Image(local_path)})
        if remove:
            os.remove(local_path)

    def write_summary(self, n_iter, pbar=None, val_set=False, remove=False):
        """Update the progress bar, and log scalars and images when wandb runs."""
        # --wandb 0 opts out even when some other code started a wandb run.
        wb = _wandb() if self.args.wandb else None
        stage = 'Val Step' if val_set else 'Step'
        if pbar is not None:
            pbar.set_description(
                f"{self.args.exp_name} | {stage}: {n_iter}, Loss: {self.j_loss.item():2.5f}")
            pbar.update(1)

        if val_set:
            if wb is not None and n_iter % self.args.log_scalar_interval == 0:
                wb.log({
                    'val_n_iter': n_iter,
                    'val_total_loss': self.j_loss.mean().item(),
                    'val_epipolar_loss_coarse': self.eloss_c.mean().item(),
                    'val_epipolar_loss_fine': self.eloss_f.mean().item(),
                    'val_cycle_loss_coarse': self.closs_c.mean().item(),
                    'val_cycle_loss_fine': self.closs_f.mean().item(),
                })
            if wb is not None and n_iter % self.args.log_img_interval == 0:
                self._log_images(n_iter, wb, 'val_', remove)
            return

        if wb is not None and n_iter % self.args.log_scalar_interval == 0:
            wb.log({
                'n_iter': n_iter,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'Total_loss': self.j_loss.item(),
                'epipolar_loss_coarse': self.eloss_c.mean().item(),
                'epipolar_loss_fine': self.eloss_f.mean().item(),
                'cycle_loss_coarse': self.closs_c.mean().item(),
                'cycle_loss_fine': self.closs_f.mean().item(),
            })

        # This visualization shows a number of query points in the first image
        # and their predicted correspondences in the second image.
        if wb is not None and n_iter % self.args.log_img_interval == 0:
            self._log_images(n_iter, wb, '', remove)

    def load_model(self, filename):
        """Load the weights, and the optimizer state only when training.

        Inference never steps the optimizer, so reading its state back would
        allocate a second copy of every parameter for nothing. After the
        weights load, it zeroes the subnormal ones, unless the model was built
        with ``zero_subnormal_weights=False``. It loads the optimizer state as
        stored.
        """
        to_load = torch.load(filename, map_location=self.device, weights_only=True)
        self.model.load_state_dict(to_load['state_dict'])
        self.zeroed_subnormal = (zero_subnormal_weights(self.model)
                                 if self.zero_subnormal_on_load else 0)
        if self.args.phase == 'train':
            if 'optimizer' in to_load.keys():
                self.optimizer.load_state_dict(to_load['optimizer'])
            if 'scheduler' in to_load.keys():
                self.scheduler.load_state_dict(to_load['scheduler'])
        return to_load['step']
    
    def _resolve_checkpoint(self):
        """Return the checkpoint to load, or None to train from scratch.

        An explicit --ckpt_path wins and must exist. Otherwise the newest .pth in
        the experiment folder resumes the run.
        """
        if self.args.ckpt_path != "":
            if not os.path.isfile(self.args.ckpt_path):
                raise Exception('no checkpoint found in the following path:{}'.format(self.args.ckpt_path))
            return self.args.ckpt_path

        ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
        os.makedirs(ckpt_folder, exist_ok=True)
        # load from the most recent ckpt from all existing ckpts
        ckpts = [os.path.join(ckpt_folder, f) for f in sorted(os.listdir(ckpt_folder))
                 if f.endswith('.pth')]
        return ckpts[-1] if ckpts else None

    def load_from_ckpt(self):
        '''
        load the checkpoint resolved in __init__ and return the current step
        :return: the current starting step
        '''
        if self.ckpt_to_load is None:
            print('No ckpts found, training from scratch...')
            return 0

        if self.args.ckpt_path != "":
            print("Reloading from {}".format(self.ckpt_to_load))
            return self.load_model(self.ckpt_to_load)

        step = self.load_model(self.ckpt_to_load)
        print('Reloading from {}, starting at step={}'.format(self.ckpt_to_load, step))
        return step

    def save_model(self, step):
        ckpt_folder = os.path.join(self.args.outdir, self.args.exp_name)
        os.makedirs(ckpt_folder, exist_ok=True)

        save_path = os.path.join(ckpt_folder, "{:06d}.pth".format(step))
        print('saving ckpts {}...'.format(save_path))
        torch.save({'step': step,
                    'state_dict': self.model.state_dict(),
                    'optimizer':  self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    },
                   save_path)