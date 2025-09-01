import math
import torch
import folder_paths


class Config:
    def __init__(self) -> None:
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.locate_head = False
        self.cxt_num = [0, 3][1]    # multi-scale skip connections from encoder
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        self.refine = ['', 'itself', 'RefUNet', 'Refiner', 'RefinerPVTInChannels4'][0]
        self.progressive_ref = self.refine and True
        self.ender = self.progressive_ref and False
        self.scale = self.progressive_ref and 2
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][2]
        self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        self.dec_blk = ['BasicDecBlk', 'ResBlk', 'HierarAttDecBlk'][0]
        self.auxiliary_classification = False
        self.refine_iteration = 1
        self.freeze_bb = False
        self.precisionHigh = True
        self.compile = True
        self.load_all = True
        self.verbose_eval = True

        self.size = 1024
        self.batch_size = 2
        self.IoU_finetune_last_epochs = [0, -20][1]     # choose 0 to skip
        if self.dec_blk == 'HierarAttDecBlk':
            self.batch_size = 2 ** [0, 1, 2, 3, 4][2]
        self.model = [
            'BiRefNet',
        ][0]

        # Components
        self.lat_blk = ['BasicLatBlk'][0]
        self.dec_channels_inter = ['fixed', 'adap'][0]

        # Backbone
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
            'pvt_v2_b2', 'pvt_v2_b5',               # 3-bs10, 4-bs5
            'swin_v1_b', 'swin_v1_l',               # 5-bs9, 6-bs6
            'swin_v1_t', 'swin_v1_s',               # 7, 8
            'pvt_v2_b0', 'pvt_v2_b1',               # 9, 10
        ][6]
        self.lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
            'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []
        # self.sys_home_dir = '/root/autodl-tmp'
        # self.weights_root_dir = os.path.join(self.sys_home_dir, 'weights')
        # self.weights = {
        #     'pvt_v2_b2': os.path.join(self.weights_root_dir, 'pvt_v2_b2.pth'),
        #     'pvt_v2_b5': os.path.join(self.weights_root_dir, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
        #     'swin_v1_b': os.path.join(self.weights_root_dir, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
        #     'swin_v1_l': os.path.join(self.weights_root_dir, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
        #     'swin_v1_t': os.path.join(self.weights_root_dir, ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0]),
        #     'swin_v1_s': os.path.join(self.weights_root_dir, ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0]),
        #     'pvt_v2_b0': os.path.join(self.weights_root_dir, ['pvt_v2_b0.pth'][0]),
        #     'pvt_v2_b1': os.path.join(self.weights_root_dir, ['pvt_v2_b1.pth'][0]),
        # }
        weight_paths_name = "birefnet"
        self.weights = {
            'pvt_v2_b2': folder_paths.get_full_path(weight_paths_name, 'pvt_v2_b2.pth'),
            'pvt_v2_b5': folder_paths.get_full_path(weight_paths_name, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
            'swin_v1_b': folder_paths.get_full_path(weight_paths_name, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
            'swin_v1_l': folder_paths.get_full_path(weight_paths_name, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
            'swin_v1_t': folder_paths.get_full_path(weight_paths_name, ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'swin_v1_s': folder_paths.get_full_path(weight_paths_name, ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'pvt_v2_b0': folder_paths.get_full_path(weight_paths_name, ['pvt_v2_b0.pth'][0]),
            'pvt_v2_b1': folder_paths.get_full_path(weight_paths_name, ['pvt_v2_b1.pth'][0]),
        }

        # Training
        self.num_workers = 5        # will be decrease to min(it, batch_size) at the initialization of the data_loader
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr = 1e-5 * math.sqrt(self.batch_size / 5)  # adapt the lr linearly
        self.lr_decay_epochs = [1e4]    # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        self.only_S_MAE = False
        self.SDPA_enabled = False    # Bug. Slower and errors occur in multi-GPUs

        # Data
        # self.data_root_dir = os.path.join(self.sys_home_dir, 'datasets/dis')
        self.task = ['DIS5K', 'COD', 'HRSOD'][0]
        self.training_set = {
            'DIS5K': 'DIS-TR',
            'COD': 'TR-COD10K+TR-CAMO',
            'HRSOD': ['TR-DUTS', 'TR-HRSOD+TR-UHRSD', 'TR-DUTS+TR-HRSOD+TR-UHRSD'][1]
        }[self.task]
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4]

        # Loss
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
            'mse': 150 * 0,         # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 10 * 1,          # help contours,
            'cnt': 5 * 0,          # help contours
        }
        self.lambdas_cls = {
            'ce': 5.0
        }
        # Adv
        self.lambda_adv_g = 10. * 0        # turn to 0 to avoid adv training
        self.lambda_adv_d = 3. * (self.lambda_adv_g > 0)

        # others
        self.device = [0, 'cpu'][0 if torch.cuda.is_available() else 1]     # .to(0) == .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7
        # run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'train.sh' == f]
        # with open(run_sh_file[0], 'r') as f:
        #     lines = f.readlines()
        #     self.save_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
        #     self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
        # self.val_step = [0, self.save_step][0]


# This code is used to decouple the code from TIMM when used for inference
if False:
    # For training
    try:
        # version > 0.6.13
        from timm.layers import DropPath, to_2tuple, trunc_normal_
    except Exception:
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
else:
    import torch.nn as nn
    from itertools import repeat
    import collections.abc

    # 1. Alias DropPath to nn.Identity for inference.
    DropPath = nn.Identity

    # 2. Dummy function for trunc_normal_ since it's not needed for inference.
    def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
        # This function is only used for initializing weights.
        # For inference, we load pre-trained weights, so this function is not needed.
        # We can simply pass.
        pass

    # 3. Copied and simplified implementation of to_2tuple.
    # From PyTorch internals
    def _ntuple(n):
        def parse(x):
            if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
                return tuple(x)
            return tuple(repeat(x, n))
        return parse

    to_2tuple = _ntuple(2)
