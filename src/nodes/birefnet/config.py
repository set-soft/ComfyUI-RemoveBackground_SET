import folder_paths
from itertools import repeat
import collections.abc


class Config:
    def __init__(self, bb_index: int = 6) -> None:
        # MODEL settings
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = [0, 3][1]    # multi-scale skip connections from encoder
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        # self.dec_att = ['', 'ASPP', 'ASPPDeformable'][2]
        # self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        # self.dec_blk = ['BasicDecBlk', 'ResBlk'][0]

        # TRAINING settings
        self.batch_size = 4

        # Backbone settings
        # Only swin_v1_l and swin_v1_t used
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
            'swin_v1_t', 'swin_v1_s',               # 3, 4
            'swin_v1_b', 'swin_v1_l',               # 5-bs9, 6-bs4
            'pvt_v2_b0', 'pvt_v2_b1',               # 7, 8
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][bb_index]
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

        # MODEL settings - inactive
        # self.lat_blk = ['BasicLatBlk'][0]
        # self.dec_channels_inter = ['fixed', 'adap'][0]
        # self.refine = ['', 'itself', 'RefUNet', 'Refiner', 'RefinerPVTInChannels4'][0]
        # self.progressive_ref = self.refine and True
        # self.ender = self.progressive_ref and False
        # self.scale = self.progressive_ref and 2
        # self.auxiliary_classification = False       # Only for DIS5K, where class labels are saved in `dataset.py`.
        # self.refine_iteration = 1
        # self.freeze_bb = False
        # self.model = 'BiRefNet'

        # PATH settings - inactive
        # https://drive.google.com/drive/folders/1cmce_emsS8A5ha5XT2c_CZiJzlLM81ms
        # self.weights_root_dir = os.path.join(self.sys_home_dir, 'weights/cv')
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

        # Callbacks - inactive
        self.SDPA_enabled = False    # Bugs. Slower and errors occur in multi-GPUs

        # others
        self.device = [0, 'cpu'][0]     # .to(0) == .to('cuda:0')


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
