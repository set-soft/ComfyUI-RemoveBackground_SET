#
# BEN (Background Erase Network) introduces a novel approach to foreground segmentation through its innovative Confidence
# Guided Matting (CGM) pipeline. The architecture employs a refiner network that targets and processes pixels where the
# base model exhibits lower confidence levels, resulting in more precise and reliable matting results.
#
# Maxwell Meyer and Jack Spruyt {maxwellmeyer, jackspruyt}@prama.llc
# arXiv:2501.06230v1 [cs.CV] 8 Jan 2025  DOI: 10.57967/hf/3503
#
# https://huggingface.co/PramaLLC/BEN
# https://huggingface.co/PramaLLC/BEN2
#
# License: MIT (was Apache 2.0)
#
# Note by Salvador E. Tropea (SET):
# I removed pre/post-processing and other stuff not related to the model.
# Also moved the SwinTransformer outside, this is the standard Swin v1 code.
# Additionally I adapted the code to support other devices and data types other than CUDA + FLOAT16
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import make_cbr, make_cbg, rescale_to, resize_as, image2patches, patches2image, MCLM, MCRM
from ..swin.swin_v1 import swin_v1_b


class BEN_Base(nn.Module):
    def __init__(self, mva_variant=False):
        super().__init__()
        if not mva_variant:
            # BEN
            act_fun = F.gelu
            act_lay = nn.GELU
            norm_tp = nn.InstanceNorm2d
        else:
            # MVANet
            act_fun = F.relu
            act_lay = nn.PReLU
            norm_tp = nn.BatchNorm2d

        self.backbone = swin_v1_b()
        emb_dim = 128
        # sideout* layers are for training, should be pruned
        self.sideout5 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout4 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout3 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        self.output5 = make_cbr(1024, emb_dim, norm_tp, act_lay)
        self.output4 = make_cbr(512, emb_dim, norm_tp, act_lay)
        self.output3 = make_cbr(256, emb_dim, norm_tp, act_lay)
        self.output2 = make_cbr(128, emb_dim, norm_tp, act_lay)
        self.output1 = make_cbr(128, emb_dim, norm_tp, act_lay)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8], act_fun=act_fun, rename12=mva_variant)
        self.conv1 = make_cbr(emb_dim, emb_dim, norm_tp, act_lay)
        self.conv2 = make_cbr(emb_dim, emb_dim, norm_tp, act_lay)
        self.conv3 = make_cbr(emb_dim, emb_dim, norm_tp, act_lay)
        self.conv4 = make_cbr(emb_dim, emb_dim, norm_tp, act_lay)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=act_fun)
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=act_fun)
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=act_fun)
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=act_fun)

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            norm_tp(384),
            act_lay(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            norm_tp(384),
            act_lay(),
            nn.Conv2d(384, emb_dim, kernel_size=3, padding=1)
        )

        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim, norm_tp)
        self.upsample2 = make_cbg(emb_dim, emb_dim, norm_tp)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, act_lay) or isinstance(m, nn.Dropout):
                m.inplace = True

    # @torch.inference_mode()
    # @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward(self, x):
        real_batch = x.size(0)

        shallow_batch = self.shallow(x)
        # Half resolution "global view"
        glb_batch = rescale_to(x, scale_factor=0.5, interpolation='bilinear')

        final_input = None
        for i in range(real_batch):
            start = i * 4
            end = (i + 1) * 4
            # Local "patch view" (4)
            loc_batch = image2patches(x[i, :, :, :].unsqueeze(dim=0))
            # 5 views: global + 4 local
            input_ = torch.cat((loc_batch, glb_batch[i, :, :, :].unsqueeze(dim=0)), dim=0)

            if final_input is None:
                final_input = input_
            else:
                final_input = torch.cat((final_input, input_), dim=0)

        features = self.backbone(final_input)
        outputs = []

        for i in range(real_batch):
            start = i * 5
            end = (i + 1) * 5

            f4 = features[4][start:end, :, :, :]  # shape: [5, C, H, W]
            f3 = features[3][start:end, :, :, :]
            f2 = features[2][start:end, :, :, :]
            f1 = features[1][start:end, :, :, :]
            f0 = features[0][start:end, :, :, :]
            e5 = self.output5(f4)  # (5,128,16,16)
            e4 = self.output4(f3)  # (5,128,32,32)
            e3 = self.output3(f2)  # (5,128,64,64)
            e2 = self.output2(f1)  # (5,128,128,128)
            e1 = self.output1(f0)  # (5,128,128,128)
            # Now separate the global from the local
            loc_e5, glb_e5 = e5.split([4, 1], dim=0)
            # Cross-attention
            e5 = self.multifieldcrossatt(loc_e5, glb_e5)  # (4,128,16,16)

            # Decoder
            e4 = self.conv4(self.dec_blk4(e4 + resize_as(e5, e4)))
            e3 = self.conv3(self.dec_blk3(e3 + resize_as(e4, e3)))
            e2 = self.conv2(self.dec_blk2(e2 + resize_as(e3, e2)))
            e1 = self.conv1(self.dec_blk1(e1 + resize_as(e2, e1)))

            loc_e1, glb_e1 = e1.split([4, 1], dim=0)

            # Recombine the patches
            output1_cat = patches2image(loc_e1)  # (1,128,256,256)

            # add glb feat in
            output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
            # merge
            final_output = self.insmask_head(output1_cat)  # (1,128,256,256)
            # shallow feature merge
            shallow = shallow_batch[i, :, :, :].unsqueeze(dim=0)
            final_output = final_output + resize_as(shallow, final_output)
            final_output = self.upsample1(rescale_to(final_output))
            final_output = rescale_to(final_output + resize_as(shallow, final_output))
            final_output = self.upsample2(final_output)
            final_output = self.output(final_output)
            mask = final_output.sigmoid()
            outputs.append(mask)

        return torch.cat(outputs, dim=0)
