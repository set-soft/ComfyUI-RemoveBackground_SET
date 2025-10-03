# Shared by MVANet
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.InstanceNorm2d(out_dim), nn.GELU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.InstanceNorm2d(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x


class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # Delay the dim_t creation until we know the target device
        # self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32)

    def __call__(self, tensor_for_reference):
        """
        Takes a reference tensor to get the batch size, height, width, dtype, and device.
        """
        device = tensor_for_reference.device
        dtype = tensor_for_reference.dtype
        b, _, h, w = tensor_for_reference.shape

        mask = torch.zeros([b, h, w], dtype=torch.bool, device=device)
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=dtype)
        x_embed = not_mask.cumsum(dim=2, dtype=dtype)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        # Here dim_t is created for the target device
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t.to(device) // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        # And the returned dtype is the same as the input tensor
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(dtype)


class MCLM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[1, 4, 8], act_fun=F.gelu, rename12=False):
        super().__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        linear1 = nn.Linear(d_model, d_model * 2)
        linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = act_fun
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        # MVANet error, they added 2, but then they removed them, but left the wrong number
        self.rename12 = rename12
        if rename12:
            self.linear5 = linear1
            self.linear6 = linear2
        else:
            self.linear1 = linear1
            self.linear2 = linear2

    def forward(self, L, g):
        """
        L: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = L.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(L, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)

        # Note: the original code used self.g_pos and self.p_poses, but initialized it on each call
        # So I just used g_pos and p_poses (SET)
        p_poses = []
        pools = []
        for pool_ratio in self.pool_ratios:
            # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))

            pos_emb = self.positional_encoding(pool)
            p_poses.append(rearrange(pos_emb, 'b c h w -> (h w) b c'))

        pools = torch.cat(pools, 0)
        p_poses = torch.cat(p_poses, dim=0)
        # Generate global positional embedding using `g` as a reference.
        g_pos = rearrange(self.positional_encoding(g), 'b c h w -> (h w) b c')

        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')

        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](g_hw_b_c + g_pos, pools + p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        linear1 = self.linear5 if self.rename12 else self.linear1
        linear2 = self.linear6 if self.rename12 else self.linear2
        g_hw_b_c = g_hw_b_c + self.dropout2(linear2(self.dropout(self.activation(linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = rearrange(L, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        L = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return rearrange(L, "(h w) b c -> b c h w", h=h, w=w)  # (5,c,h*w)


class MCRM(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None, act_fun=F.gelu):
        super().__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = act_fun
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios

    def forward(self, x):
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w

        # b(4),c,h,w
        patched_glb = rearrange(glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        # generate token attention map
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map, size=patches2image(loc).shape[-2:], mode='nearest')
        loc = loc * rearrange(token_attention_map, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool, 'nl c h w -> nl c (h w)'))  # nl(4),c,hw

        # nl(4),c,nphw -> nl(4),nphw,1,c
        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')

        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):  # traverse all local patches
            # np*hw,1,c
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])

        outputs = torch.cat(outputs, 1)
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode='nearest')  # freshed glb

        return torch.cat((src, glb), 0)  # , token_attention_map
