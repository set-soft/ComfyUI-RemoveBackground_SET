#
# Multi-view Aggregation Network for Dichotomous Image Segmentation
# Qian Yu, Xiaoqi Zhao, Youwei Pang, Lihe Zhang, Huchuan Lu
# https://arxiv.org/abs/2404.07445
#
# https://github.com/qianyu-dlut/MVANet/
#
# License: MIT
#
# Note by Salvador E. Tropea (SET):
# - I removed the training version of the model
# - The code is clearly based on BEN, so I made it share a lot
#
import torch
import torch.nn.functional as F
from torch import nn

from ..swin.swin_v1 import swin_v1_b
from .util import make_cbr, make_cbg
from ..ben.util import rescale_to, resize_as, image2patches, patches2image, MCLM, MCRM


# model for multi-scale testing
class MVANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = swin_v1_b()

        emb_dim = 128
        self.output5 = make_cbr(1024, emb_dim)
        self.output4 = make_cbr(512, emb_dim)
        self.output3 = make_cbr(256, emb_dim)
        self.output2 = make_cbr(128, emb_dim)
        self.output1 = make_cbr(128, emb_dim)

        self.multifieldcrossatt = MCLM(emb_dim, 1, [1, 4, 8], act_fun=F.relu, rename12=True)
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.conv2 = make_cbr(emb_dim, emb_dim)
        self.conv3 = make_cbr(emb_dim, emb_dim)
        self.conv4 = make_cbr(emb_dim, emb_dim)
        self.dec_blk1 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=F.relu)
        self.dec_blk2 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=F.relu)
        self.dec_blk3 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=F.relu)
        self.dec_blk4 = MCRM(emb_dim, 1, [2, 4, 8], act_fun=F.relu)

        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, emb_dim, kernel_size=3, padding=1)
        )

        self.shallow = nn.Sequential(nn.Conv2d(3, emb_dim, kernel_size=3, padding=1))
        self.upsample1 = make_cbg(emb_dim, emb_dim)
        self.upsample2 = make_cbg(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x):
        shallow = self.shallow(x)
        glb = rescale_to(x, scale_factor=0.5, interpolation='bilinear')
        loc = image2patches(x)
        input = torch.cat((loc, glb), dim=0)
        feature = self.backbone(input)
        e5 = self.output5(feature[4])
        e4 = self.output4(feature[3])
        e3 = self.output3(feature[2])
        e2 = self.output2(feature[1])
        e1 = self.output1(feature[0])
        loc_e5, glb_e5 = e5.split([4, 1], dim=0)
        e5_cat = self.multifieldcrossatt(loc_e5, glb_e5)

        e4 = self.conv4(self.dec_blk4(e4 + resize_as(e5_cat, e4)))
        e3 = self.conv3(self.dec_blk3(e3 + resize_as(e4, e3)))
        e2 = self.conv2(self.dec_blk2(e2 + resize_as(e3, e2)))
        e1 = self.conv1(self.dec_blk1(e1 + resize_as(e2, e1)))
        loc_e1, glb_e1 = e1.split([4, 1], dim=0)
        # after decoder, concat loc features to a whole one, and merge
        output1_cat = patches2image(loc_e1)
        # add glb feat in
        output1_cat = output1_cat + resize_as(glb_e1, output1_cat)
        # merge
        final_output = self.insmask_head(output1_cat)
        # shallow feature merge
        final_output = final_output + resize_as(shallow, final_output)
        final_output = self.upsample1(rescale_to(final_output))
        final_output = rescale_to(final_output + resize_as(shallow, final_output))
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)

        return final_output.sigmoid()
