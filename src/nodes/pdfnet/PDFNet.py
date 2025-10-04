#
# High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy
# Prior of Depth Fusion Network (PDFNet)
# Xianjie Liu, Keren Fu, Qijun Zhao
# https://arxiv.org/abs/2503.06100
#
# https://github.com/Tennine2077/PDFNet
# https://huggingface.co/spaces/Tennineee/PDFNet/tree/main
#
# License: MIT
#
# Note by Salvador E. Tropea (SET):
# I removed training code and made use of my copy of Swin
# Also removed the use of CLI args, the parameters from there are only for training or tied to SwinB geometry
#
from einops import rearrange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import RMSNorm, SwiGLU  # Meta common layers
from ..swin.swin_v1 import swin_v1_b


def make_crs(in_dim, out_dim):
    """ Convolution + RMSNorm + SiLU == CRS """
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         RMSNorm(out_dim),
                         nn.SiLU(inplace=True))


def rescale_to(src, size):  # _upscale_
    return F.upsample(src, size=size, mode='bilinear', align_corners=True)


def resize_as(src, tar):  # _upscale_like
    return F.upsample(src, size=tar.shape[2:], mode='bilinear', align_corners=True)


class PDF_depth_decoder(nn.Module):
    def __init__(self, raw_ch=3, out_ch=1, emb_dim=128):
        super().__init__()

        self.Decoder = nn.ModuleList()
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2, emb_dim*2), make_crs(emb_dim*2, emb_dim)))
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2, emb_dim*2), make_crs(emb_dim*2, emb_dim)))
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2, emb_dim*2), make_crs(emb_dim*2, emb_dim)))
        self.Decoder.append(nn.Sequential(make_crs(emb_dim*2, emb_dim*2), make_crs(emb_dim*2, emb_dim)))

        self.shallow = nn.Sequential(nn.Conv2d(raw_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.upsample1 = make_crs(emb_dim, emb_dim)
        self.upsample2 = make_crs(emb_dim, emb_dim)

        self.Bside = nn.ModuleList()
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))

    def forward(self, img, img_feature):

        L1_feature, L2_feature, L3_feature, L4_feature, global_feature = img_feature

        De_L4 = self.Decoder[0](torch.cat([global_feature, L4_feature], dim=1))
        De_L3 = self.Decoder[1](torch.cat([resize_as(De_L4, L3_feature), L3_feature], dim=1))
        De_L2 = self.Decoder[2](torch.cat([resize_as(De_L3, L2_feature), L2_feature], dim=1))
        De_L1 = self.Decoder[3](torch.cat([resize_as(De_L2, L1_feature), L1_feature], dim=1))

        shallow = self.shallow(img)
        final_output = De_L1 + resize_as(shallow, De_L1)
        final_output = self.upsample1(rescale_to(final_output, [final_output.shape[-2]*2, final_output.shape[-1]*2]))
        final_output = rescale_to(final_output + resize_as(shallow, final_output), [final_output.shape[-2]*2,
                                  final_output.shape[-1]*2])
        final_output = self.upsample2(final_output)

        final_output = self.Bside[0](final_output)

        side_1 = self.Bside[1](De_L1)
        side_2 = self.Bside[2](De_L2)
        side_3 = self.Bside[3](De_L3)
        side_4 = self.Bside[4](De_L4)

        return [final_output, side_1, side_2, side_3, side_4]


class CoA(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.Att = nn.MultiheadAttention(emb_dim, 1, bias=False, batch_first=True, dropout=0.1)
        self.Norm1 = RMSNorm(emb_dim, data_format='channels_last')
        self.drop1 = nn.Dropout(0.1)
        self.FFN = SwiGLU(emb_dim, emb_dim)
        self.Norm2 = RMSNorm(emb_dim, data_format='channels_last')
        self.drop2 = nn.Dropout(0.1)

    def forward(self, q, kv):
        res = q
        KV_feature = self.Att(q, kv, kv)[0]
        KV_feature = self.Norm1(self.drop1(KV_feature)) + res
        res = KV_feature
        KV_feature = self.FFN(KV_feature)
        KV_feature = self.Norm2(self.drop2(KV_feature)) + res
        return KV_feature


class FSE(nn.Module):
    def __init__(self, img_dim=128, depth_dim=128, patch_dim=128, emb_dim=128, pool_ratio=[1, 1, 1], patch_ratio=4):
        super().__init__()

        self.patch_ratio = patch_ratio
        self.pool_ratio = pool_ratio
        self.I_channelswich = make_crs(img_dim, emb_dim)
        self.P_channelswich = make_crs(patch_dim, emb_dim)
        self.D_channelswich = make_crs(depth_dim, emb_dim)

        self.IP = CoA(emb_dim)
        self.PI = CoA(emb_dim)

        self.ID = CoA(emb_dim)
        self.DI = CoA(emb_dim)

    def split(self, x: torch.Tensor, patch_ratio: int = 8) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        B, C, H, W = x.shape
        patch_stride = H//patch_ratio  # int(patch_size * (1 - overlap_ratio))
        patch_size = H//patch_ratio
        steps = patch_ratio

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx: batch_size * (idx + 1)]
                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def get_boundary(self, pred):
        if pred.shape[-2]//8 % 2 == 0:
            return abs(pred.sigmoid() - F.avg_pool2d(pred.sigmoid(),
                                                     kernel_size=(pred.shape[-2]//8+1, pred.shape[-1]//8+1),
                                                     stride=1,
                                                     padding=(pred.shape[-2]//8//2, pred.shape[-1]//8//2)))
        else:
            return abs(pred.sigmoid() - F.avg_pool2d(pred.sigmoid(),
                                                     kernel_size=(pred.shape[-2]//8, pred.shape[-1]//8),
                                                     stride=1,
                                                     padding=(pred.shape[-2]//8//2, pred.shape[-1]//8//2)))

    def BIS(self, pred):
        if pred.shape[-2]//8 % 2 == 0:
            boundary = 2*self.get_boundary(pred.sigmoid())
            return boundary, F.relu(pred.sigmoid()-5*boundary)
        else:
            boundary = 2*self.get_boundary(pred.sigmoid())
            return boundary, F.relu(pred.sigmoid()-5*boundary)

    def forward(self, img, depth, patch, last_pred):
        boundary, integrity = self.BIS(last_pred)
        img = img * resize_as(last_pred.sigmoid(), img)
        depth = depth * resize_as(last_pred.sigmoid(), depth)
        patch = patch * resize_as(last_pred.sigmoid(), patch)
        pi, pd, pp = self.pool_ratio
        B, C, img_H, img_W = img.size()
        img_cs = self.I_channelswich(img)
        pool_img_cs = F.adaptive_avg_pool2d(img_cs, output_size=[img_H//pi, img_W//pi])
        img_cs = rearrange(img_cs, 'b c h w -> b (h w) c')
        pool_img_cs = rearrange(pool_img_cs, 'b c h w -> b (h w) c')
        B, C, depth_H, depth_W = depth.size()

        # give depth the integrity prior
        integrity = resize_as(integrity, depth)
        last_pred_sigmoid = resize_as(last_pred, depth).sigmoid()
        enhance_depth = depth*(last_pred_sigmoid + integrity)
        depth_cs = self.D_channelswich(enhance_depth)
        pool_depth_cs = F.adaptive_avg_pool2d(depth_cs, output_size=[depth_H//pd, depth_W//pd])
        pool_depth_cs = rearrange(pool_depth_cs, 'b c h w -> b (h w) c')
        B, C, patch_H, patch_W = patch.size()

        # select the boundary patches to select patches
        patch_batch = self.split(patch, patch_ratio=self.patch_ratio)
        boundary_batch = self.split(boundary, patch_ratio=self.patch_ratio)
        boundary_score = boundary_batch.mean(dim=[2, 3])[..., None, None]
        select_patch = patch_batch * (1+5*boundary_score)
        select_patch = self.merge(select_patch, batch_size=B)

        patch_cs = self.P_channelswich(select_patch)
        pool_patch_cs = F.adaptive_avg_pool2d(patch_cs, output_size=[patch_H//pp, patch_W//pp])
        pool_patch_cs = rearrange(pool_patch_cs, 'b c h w -> b (h w) c')

        patch_feature = self.PI(pool_patch_cs, torch.cat([pool_img_cs, pool_depth_cs], dim=1))
        img_feature = self.IP(img_cs, patch_feature)

        depth_feature = self.DI(pool_depth_cs, torch.cat([pool_img_cs, pool_patch_cs], dim=1))
        img_feature = self.ID(img_feature, depth_feature)

        patch_feature = rearrange(patch_feature, 'b (h w) c -> b c h w', h=patch_H//pp)
        depth_feature = rearrange(depth_feature, 'b (h w) c -> b c h w', h=depth_H//pd)
        img_feature = rearrange(img_feature, 'b (h w) c -> b c h w', h=img_H)

        depth_feature = resize_as(depth_feature, depth)
        patch_feature = resize_as(patch_feature, patch)

        return img_feature + rearrange(img_cs, 'b (h w) c -> b c h w', h=img_H), \
            depth_feature + depth_cs, patch_feature + patch_cs


class PDF_decoder(nn.Module):
    def __init__(self, raw_ch=3, out_ch=1, emb_dim=128):
        super().__init__()
        self.patch_ratio = 8

        self.FSE_mix = nn.ModuleList()
        self.FSE_mix.append(FSE(emb_dim*2, emb_dim*2, emb_dim*2, emb_dim, pool_ratio=[1, 1, 1], patch_ratio=self.patch_ratio))
        self.FSE_mix.append(FSE(emb_dim*2, emb_dim*2, emb_dim*2, emb_dim, pool_ratio=[1, 1, 1], patch_ratio=self.patch_ratio))
        self.FSE_mix.append(FSE(emb_dim*2, emb_dim*2, emb_dim*2, emb_dim, pool_ratio=[2, 2, 2], patch_ratio=self.patch_ratio))
        self.FSE_mix.append(FSE(emb_dim*2, emb_dim*2, emb_dim*2, emb_dim, pool_ratio=[2, 2, 2], patch_ratio=self.patch_ratio))

        self.shallow = nn.Sequential(nn.Conv2d(raw_ch*2, emb_dim, kernel_size=4, stride=4), make_crs(emb_dim, emb_dim))
        self.upsample1 = nn.Sequential(make_crs(emb_dim, emb_dim))
        self.upsample2 = nn.Sequential(make_crs(emb_dim, emb_dim))

        self.channel_mix = nn.ModuleList()
        self.channel_mix.append(make_crs(emb_dim*3, emb_dim))
        self.channel_mix.append(make_crs(emb_dim*3, emb_dim))
        self.channel_mix.append(make_crs(emb_dim*3, emb_dim))
        self.channel_mix.append(make_crs(emb_dim*3, emb_dim))

        self.Bside = nn.ModuleList()
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))
        self.Bside.append(nn.Conv2d(emb_dim, out_ch, 3, padding=1))

    def forward(self, img, depth, img_feature, depth_feature, patch_img_feature):
        B, C, H, W = img.size()
        side_5 = self.Bside[5](resize_as(img_feature[4], patch_img_feature[4]) +
                               resize_as(depth_feature[4], patch_img_feature[4]) +
                               patch_img_feature[4])

        img_L4, depth_L4, patch_L4 = self.FSE_mix[0](
            torch.cat([img_feature[4], img_feature[3]], dim=1),
            torch.cat([depth_feature[4], depth_feature[3]], dim=1),
            torch.cat([patch_img_feature[4], patch_img_feature[3]], dim=1), side_5
        )
        mix_L4 = self.channel_mix[3](torch.cat([resize_as(img_L4, patch_L4), resize_as(depth_L4, patch_L4),
                                                patch_L4], dim=1))
        side_4 = self.Bside[4](mix_L4)
        img_L3, depth_L3, patch_L3 = self.FSE_mix[1](
            torch.cat([resize_as(img_L4, img_feature[2]), img_feature[2]], dim=1),
            torch.cat([resize_as(depth_L4, depth_feature[2]), depth_feature[2]], dim=1),
            torch.cat([resize_as(patch_L4, patch_img_feature[2]), patch_img_feature[2]], dim=1), side_4
        )
        mix_L3 = self.channel_mix[2](torch.cat([resize_as(img_L3, patch_L3), resize_as(depth_L3, patch_L3),
                                                patch_L3], dim=1))
        side_3 = self.Bside[3](mix_L3)
        img_L2, depth_L2, patch_L2 = self.FSE_mix[2](
            torch.cat([resize_as(img_L3, img_feature[1]), img_feature[1]], dim=1),
            torch.cat([resize_as(depth_L3, depth_feature[1]), depth_feature[1]], dim=1),
            torch.cat([resize_as(patch_L3, patch_img_feature[1]), patch_img_feature[1]], dim=1), side_3
        )
        mix_L2 = self.channel_mix[1](torch.cat([resize_as(img_L2, patch_L2), resize_as(depth_L2, patch_L2),
                                                patch_L2], dim=1))
        side_2 = self.Bside[2](mix_L2)
        img_L1, depth_L1, patch_L1 = self.FSE_mix[3](
            torch.cat([resize_as(img_L2, img_feature[0]), img_feature[0]], dim=1),
            torch.cat([resize_as(depth_L2, depth_feature[0]), depth_feature[0]], dim=1),
            torch.cat([resize_as(patch_L2, patch_img_feature[0]), patch_img_feature[0]], dim=1), side_2
        )
        mix_L1 = self.channel_mix[0](torch.cat([resize_as(img_L1, patch_L1), resize_as(depth_L1, patch_L1),
                                                patch_L1], dim=1))
        side_1 = self.Bside[1](mix_L1)

        shallow = self.shallow(rescale_to(torch.cat([img, depth], dim=1), [H*4, W*4]))
        final_output = rescale_to(mix_L1, [mix_L1.shape[-2]*2, mix_L1.shape[-1]*2]) + \
            rescale_to(shallow, [mix_L1.shape[-2]*2, mix_L1.shape[-1]*2])
        final_output = self.upsample1(final_output)
        final_output = rescale_to(final_output, [final_output.shape[-2]*2, final_output.shape[-1]*2]) + shallow
        final_output = self.upsample2(final_output)

        final_output = self.Bside[0](final_output)

        return [final_output, side_1, side_2, side_3, side_4, side_5]


class PDFNet_process(nn.Module):
    def __init__(self, encoder, decoder, depth_decoder, emb_dim=128, channels=[128, 256, 512, 1024]):
        super().__init__()
        self.patch_ratio = 8
        self.raw_ch = 3
        self.Glob = nn.Sequential(make_crs(emb_dim, emb_dim))
        self.decoder = decoder
        self.depth_decoder = depth_decoder
        self.decoder.patch_ratio = self.patch_ratio

        self.channel_mix = make_crs(emb_dim*4, emb_dim)
        self.channel_mix4 = make_crs(channels[3], emb_dim)
        self.channel_mix3 = make_crs(channels[2], emb_dim)
        self.channel_mix2 = make_crs(channels[1], emb_dim)
        self.channel_mix1 = make_crs(channels[0], emb_dim)

        self.encoder = encoder

    def encode(self, x, encoder):
        latent_I1, latent_I2, latent_I3, latent_I4 = encoder(x)

        latent_I1 = self.channel_mix1(latent_I1)
        latent_I2 = self.channel_mix2(latent_I2)
        latent_I3 = self.channel_mix3(latent_I3)
        latent_I4 = self.channel_mix4(latent_I4)
        x_glob = self.Glob(self.channel_mix(torch.cat([resize_as(latent_I1, latent_I4),
                                                       resize_as(latent_I2, latent_I4),
                                                       resize_as(latent_I3, latent_I4),
                                                       latent_I4], dim=1)))

        return latent_I1, latent_I2, latent_I3, latent_I4, x_glob

    def split(self, x: torch.Tensor, patch_size: int = 256, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx: batch_size * (idx + 1)]

                if padding > 0:
                    if j != 0:
                        output = output[..., padding:, :]
                    if i != 0:
                        output = output[..., :, padding:]
                    if j != steps - 1:
                        output = output[..., :-padding, :]
                    if i != steps - 1:
                        output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def forward(self, img, depth):
        # Normalize the depth map to [0,1]
        depth = (depth-depth.min())/(depth.max()-depth.min())

        B, C, H, W = img.size()
        RIMG, RDEPTH = img, depth
        if RDEPTH.shape[1] == 1:
            RDEPTH = RDEPTH.repeat(1, 3, 1, 1)

        down_ratio = 2
        patch_ratio = self.patch_ratio
        Down_RIMG = rescale_to(RIMG, [RIMG.shape[-2]//down_ratio, RIMG.shape[-1]//down_ratio])
        Down_RDEPTH = rescale_to(RDEPTH, [RDEPTH.shape[-2]//down_ratio, RDEPTH.shape[-1]//down_ratio])
        Down_img_depth = torch.cat([Down_RIMG, Down_RDEPTH], dim=0)

        # Encode the image and the depth
        latent_I1, latent_I2, latent_I3, latent_I4, x_glob = self.encode(Down_img_depth, self.encoder)

        # Separate the encoded image and depth
        Depth_latent_I1 = latent_I1[B:2*B]
        Depth_latent_I2 = latent_I2[B:2*B]
        Depth_latent_I3 = latent_I3[B:2*B]
        Depth_latent_I4 = latent_I4[B:2*B]
        Depth_x_glob = x_glob[B:2*B]

        latent_I1 = latent_I1[:B]
        latent_I2 = latent_I2[:B]
        latent_I3 = latent_I3[:B]
        latent_I4 = latent_I4[:B]
        x_glob = x_glob[:B]

        # Encode the splitted image
        patch_img = self.split(RIMG, patch_size=RIMG.shape[-2]//patch_ratio, overlap_ratio=0.)
        patch_latent_I1, patch_latent_I2, patch_latent_I3, patch_latent_I4, patch_x_glob = self.encode(patch_img, self.encoder)

        # Merge the patches
        patch_latent_I1 = self.merge(patch_latent_I1, batch_size=B, padding=0)
        patch_latent_I2 = self.merge(patch_latent_I2, batch_size=B, padding=0)
        patch_latent_I3 = self.merge(patch_latent_I3, batch_size=B, padding=0)
        patch_latent_I4 = self.merge(patch_latent_I4, batch_size=B, padding=0)
        patch_x_glob = self.merge(patch_x_glob, batch_size=B, padding=0)

        # Decode
        pred_m = self.decoder(RIMG, RDEPTH,
                              [latent_I1, latent_I2, latent_I3, latent_I4, x_glob],
                              [Depth_latent_I1, Depth_latent_I2, Depth_latent_I3, Depth_latent_I4, Depth_x_glob],
                              [patch_latent_I1, patch_latent_I2, patch_latent_I3, patch_latent_I4, patch_x_glob])

        return pred_m[0].sigmoid()  # , pred_m[0]


def build_model(backbone_arch='swin_v1_b'):
    assert backbone_arch == 'swin_v1_b'
    # PDFNet_swinB
    # SwinB sizes
    emb_dim = 128
    channels = [128, 256, 512, 1024]
    return PDFNet_process(encoder=swin_v1_b(ape=True, full_output=False),
                          decoder=PDF_decoder(emb_dim=emb_dim),
                          depth_decoder=PDF_depth_decoder(emb_dim=emb_dim),
                          emb_dim=emb_dim, channels=channels)
