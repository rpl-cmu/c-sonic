"""Train C-SONIC on pairs of sonar images."""
import os

import torch
from tqdm import tqdm

import config
from CSONIC.csonic_model import CSONICModel
from dataloader.sonardata import SonarDataLoader
from utils import cycle


def init_wandb(args):
    """Start a wandb run when --wandb is set. Returns None otherwise.

    Authenticate outside this script: set WANDB_API_KEY or run `wandb login`
    one time. The commented login line below is a placeholder only. Keep real
    keys out of the repository.
    """
    if not args.wandb:
        return None
    import wandb
    # wandb.login(key="YOUR_WANDB_API_KEY")
    return wandb.init(project=args.wandb_project, name=args.exp_name,
                      config=vars(args))


def train_sonardata(args):
    """Run the training loop, validating and saving every --save_interval steps."""
    # save a copy of the current args in the output folder
    out_folder = os.path.join(args.outdir, args.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, 'args.txt'), 'w') as file:
        for arg in vars(args):
            file.write('{} = {}\n'.format(arg, getattr(args, arg)))

    # train data loader
    train_loader = SonarDataLoader(args).load_data()
    train_loader_iterator = iter(cycle(train_loader))

    # val data loader
    args.phase = 'val'
    val_loader = SonarDataLoader(args).load_data()
    val_loader_iterator = iter(cycle(val_loader))
    args.phase = 'train'

    # define model
    model = CSONICModel(args)
    start_step = model.start_step

    # val log iteration
    val_log_i = 0

    # training loop
    pbar = tqdm(total=args.n_iters)
    for step in range(start_step + 1, start_step + args.n_iters + 1):
        if not model.model.training:
            model.model.train()
        data = next(train_loader_iterator)
        model.set_input(data)
        model.optimize_parameters()
        model.write_summary(step, pbar)
        if step % args.save_interval == 0 and step > 0:
            model.save_model(step)
            # run val loop
            avg_loss = validate_model(model, args, val_log_i,
                                      val_loader_iterator=val_loader_iterator,
                                      pbar=pbar)
            print('step {}: average validation loss {:.5f}'.format(step, avg_loss))
            val_log_i = val_log_i + args.n_val_iters
    pbar.close()


def validate_model(model, args, val_log_i, val_loader_iterator, pbar=None):
    """Run --n_val_iters validation batches. Returns the average total loss."""
    model.model.eval()
    start_step = val_log_i
    total_loss = 0.0
    num_steps = 0
    for step in range(start_step + 1, start_step + args.n_val_iters + 1):
        data = next(val_loader_iterator)
        with torch.no_grad():
            model.set_input(data)
            loss = model.val_forward()
            total_loss += loss[0].item()
            num_steps += 1
            model.write_summary(step, pbar, val_set=True)

    model.model.train()
    return total_loss / num_steps


if __name__ == '__main__':
    args = config.get_args()
    init_wandb(args)
    train_sonardata(args)
