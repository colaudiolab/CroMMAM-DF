from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        
        
        # network saving and loading parameters
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=0, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=40, help='number of epochs to linearly decay learning rate to zero (max epochs)')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=40, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--lr_weight_decay', type=float, default=1e-7, help='learning rate decay')

        parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'], type=str, help='masked strategy of video tokens/patches')
        parser.add_argument('--mask_ratio', default=0.5, type=float, help='ratio of the visual tokens/patches need be masked')

        # finetune parameters
        parser.add_argument('--isFinetune', type=bool, default=False, help='whether to finetune')
        parser.add_argument('--finetune_lr', type=float, default=1e-6, help='initial learning rate for finetune task (1/10 of lr)')
        parser.add_argument('--finetune_lr_weight_decay', type=float, default=1e-8, help='finetune learning rate decay')
        parser.add_argument('--load_model_suffix', type=str, default='', help='the suffix of the model to be loaded')
        parser.add_argument('--load_dir', type=str, default=None, help='the dir of model to be loaded')
        self.isTrain = True
        return parser
