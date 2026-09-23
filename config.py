import configargparse


def get_args(argv=None):
    """Parse the command line.

    Pass a list of strings as ``argv`` to parse it instead of ``sys.argv``. An
    empty list gives the defaults, which is what notebooks and tests want.
    """
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True, help='config file path')

    ## path options
    parser.add_argument('--datadir', type=str, default='',
                        help='the training dataset directory')
    parser.add_argument('--val_data_dir', type=str, default='',
                        help='the validation dataset directory; '
                             'falls back to --datadir when empty')
    parser.add_argument('--outdir', type=str, default='./checkpoints/',
                        help='dir of output e.g., ckpts')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='specific checkpoint path to load the model from, '
                             'if not specified, automatically reload from most recent checkpoints')

    ## pair lists
    # Every path below is read relative to the dataset root. An absolute path is
    # used as given.
    parser.add_argument('--pairs_file', type=str, default='logs/pairs.txt',
                        help='training image pairs, one "image1 image2" per line')
    parser.add_argument('--pairs_pos_file', type=str, default='logs/pairs_pos.txt',
                        help='training pose pairs, in the same order as --pairs_file')
    parser.add_argument('--pairs_meta_file', type=str, default='logs/pairs_meta.txt',
                        help='training metadata pairs, in the same order as --pairs_file')
    parser.add_argument('--val_pairs_file', type=str, default='logs/pairs_val.txt',
                        help='validation image pairs')
    parser.add_argument('--val_pairs_pos_file', type=str, default='logs/pairs_pos_val.txt',
                        help='validation pose pairs, in the same order as --val_pairs_file')
    parser.add_argument('--val_pairs_meta_file', type=str, default='logs/pairs_meta_val.txt',
                        help='validation metadata pairs, in the same order as --val_pairs_file')

    ## general options
    parser.add_argument('--exp_name', type=str, default='csonic_resnet34_64dim',
                        help='experiment name')
    parser.add_argument('--n_iters', type=int, default=500000,
                        help='max number of training iterations')
    parser.add_argument('--phase', type=str, default='train', help='train/val/test')

    ## data options
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--num_pts', type=int, default=50,
                        help='num of points trained in each pair')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='num of samples for each arc')
    parser.add_argument('--akaze_superpoint_pts', type=int, default=20,
                        help='minimum keypoints required from each detector')
    parser.add_argument('--superpoint_weights', type=str,
                        default='pretrained/superpoint_v1.pth',
                        help='path to the SuperPoint weights')

    ## training options
    parser.add_argument('--batch_size', type=int, default=14, help='input batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='base learning rate')
    parser.add_argument('--lrate_decay_steps', type=int, default=30000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--lrate_decay_factor', type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 penalty)')

    ## model options
    parser.add_argument('--backbone', type=str, default='resnet34',
                        help='backbone for feature representation extraction. supported: resnet')
    parser.add_argument('--pretrained', type=int, default=1,
                        help='if use ImageNet pretrained weights to initialize the network')
    parser.add_argument('--coarse_feat_dim', type=int, default=64,
                        help='the feature dimension for coarse level features')
    parser.add_argument('--fine_feat_dim', type=int, default=64,
                        help='the feature dimension for fine level features')
    parser.add_argument('--prob_from', type=str, default='correlation',
                        help='compute prob by softmax(correlation score), or softmax(-distance),'
                             'options: correlation|distance')
    parser.add_argument('--window_size', type=float, default=0.125,
                        help='the size of the window, w.r.t image width at the fine level')
    parser.add_argument('--use_nn', type=int, default=1,
                        help='if use nearest neighbor in the coarse level')
    parser.add_argument('--cross_attention', type=int, default=1,
                        help='1 = C-SONIC cross-attention, 0 = SONIC baseline')

    ## loss function options
    parser.add_argument('--w_epipolar_coarse', type=float, default=9,
                        help='coarse level epipolar loss weight')
    parser.add_argument('--w_epipolar_fine', type=float, default=7,
                        help='fine level epipolar loss weight')
    parser.add_argument('--w_cycle_coarse', type=float, default=1,
                        help='coarse level cycle consistency loss weight')
    parser.add_argument('--w_cycle_fine', type=float, default=1,
                        help='fine level cycle consistency loss weight. A typo in the '
                             'old code reused the coarse weight here, so the released '
                             'model trained with 1; the default keeps that value')

    ## logging options
    parser.add_argument('--log_scalar_interval', type=int, default=20, help='print interval')
    parser.add_argument('--log_img_interval', type=int, default=2000, help='log image interval')
    parser.add_argument('--save_interval', type=int, default=20000,
                        help='frequency of weight ckpt saving')
    parser.add_argument('--num_kpts_shown', type=int, default=25,
                        help='number of keypoints shown in log images')
    parser.add_argument('--wandb', type=int, default=0, help='1 = log the run to wandb')
    parser.add_argument('--wandb_project', type=str, default='c-sonic',
                        help='wandb project name')

    ## val options
    parser.add_argument('--n_val_iters', type=int, default=4303,
                        help='max number of val iterations, no_val_pairs/batch_size')

    return parser.parse_args(argv)
