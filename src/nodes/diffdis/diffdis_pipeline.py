from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel
import torch.nn.functional as F

from ..diffusers.unet_2d_blocks import get_down_block, get_mid_block, get_up_block
from ..diffusers.embeddings import Timesteps, TimestepEmbedding
# Some constants
RESNET_GROUPS = 32
RESNET_EPS = 1e-5
RESNET_ACT_FN = "silu"
IMG_SIZE = 1024
# SAMPLE_SIZE = 96
IN_CHANNELS = 8
OUT_CHANNELS = 4
PROJECTION_CLASS_EMBEDDINGS_INPUT_DIM = 4
LATENT_SCALE_FACTOR = 0.18215
RESCALE_MODE = 'bilinear'


def resize(img, size):
    return F.interpolate(img, size=size, mode=RESCALE_MODE, align_corners=False)


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU())


def with_cross_att(block):
    return hasattr(block, "has_cross_attention") and block.has_cross_attention


class DiffDIS(nn.Module):
    def __init__(self):
        super().__init__()
        # block_out_channels
        boc0 = 320
        boc1 = boc0 * 2
        boc2 = boc1 * 2
        boc3 = boc2
        block_out_channels = [boc0, boc1, boc2, boc3]

        # SWCI
        self.rgb_proj = nn.ModuleList([nn.Conv2d(boc0, boc0, kernel_size=3, padding=1),
                                       nn.Conv2d(boc1, boc1, kernel_size=3, padding=1),
                                       nn.Conv2d(boc2, boc2, kernel_size=3, padding=1)])
        self.rgb_proj1 = nn.ModuleList([nn.Conv2d(boc0, boc0, kernel_size=3, padding=1),
                                        nn.Conv2d(boc1, boc1, kernel_size=3, padding=1),
                                        nn.Conv2d(boc2, boc2, kernel_size=3, padding=1)])
        self.rgb_conv = nn.ModuleList([make_cbg(4, boc0), make_cbg(4, boc1), make_cbg(4, boc2)])

        attention_head_dim = num_attention_heads = [5, 10, 20, 20]

        # input
        self.conv_in = nn.Conv2d(IN_CHANNELS, boc0, kernel_size=3, padding=1)

        # time
        time_embed_dim = boc0 * 4
        self.time_proj = Timesteps(boc0)
        self.time_embedding = TimestepEmbedding(boc0, time_embed_dim)

        # class embedding
        self.class_embedding = TimestepEmbedding(PROJECTION_CLASS_EMBEDDINGS_INPUT_DIM, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = boc0
        for i, down_block_type in enumerate(("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                                             "DownBlock2D")):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=2,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=RESNET_EPS,
                resnet_act_fn=RESNET_ACT_FN,
                resnet_groups=RESNET_GROUPS,
                cross_attention_dim=IMG_SIZE,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                use_linear_projection=True,
                attention_head_dim=attention_head_dim[i],
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            "UNetMidBlock2DCrossAttn",
            temb_channels=time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=RESNET_EPS,
            resnet_act_fn=RESNET_ACT_FN,
            resnet_groups=RESNET_GROUPS,
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=IMG_SIZE,
            use_linear_projection=True,
            attention_head_dim=attention_head_dim[-1],
            mid_extra_cross=True,
            mode='DBIA',
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            up_block = get_up_block(
                up_block_type,
                num_layers=3,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=i != len(block_out_channels) - 1,  # add upsample block for all BUT final layer
                resnet_eps=RESNET_EPS,
                resnet_act_fn=RESNET_ACT_FN,
                resolution_idx=i,
                resnet_groups=RESNET_GROUPS,
                cross_attention_dim=IMG_SIZE,
                num_attention_heads=reversed_num_attention_heads[i],
                use_linear_projection=True,
                attention_head_dim=attention_head_dim[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=RESNET_GROUPS, eps=RESNET_EPS)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], OUT_CHANNELS, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timesteps: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        class_labels: torch.Tensor,
        rgb_token: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        r"""
        The [`DiffDIS`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor`): The timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`):
                Class labels for conditioning. Their embeddings will be summed with the timestep embeddings.

        Returns:
            The sample tensor.
        """
        # Note: we always use 1024x1024 images, so no upscale needed

        # 1. time
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb)
        class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
        emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for j, downsample_block in enumerate(self.down_blocks):
            if with_cross_att(downsample_block):  # First 3 are CrossAttnDownBlock2D
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb,
                                                       encoder_hidden_states=encoder_hidden_states)
            else:  # The last is DownBlock2D
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

            if j < 3:
                rgb_latents = self.rgb_conv[j](rgb_token[j+1])
                rgb_latents_zero0 = self.rgb_proj[j](rgb_latents)
                rgb_latents_zero1 = self.rgb_proj1[j](rgb_latents)

                down_block_res_samples = list(down_block_res_samples)
                down_block_res_samples[-1] = (((down_block_res_samples[-1] * rgb_latents_zero0) + rgb_latents_zero1) +
                                              down_block_res_samples[-1])
                down_block_res_samples = tuple(down_block_res_samples)

        # 4. mid (UNetMidBlock2DCrossAttn)
        sample = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if with_cross_att(upsample_block):  # 2nd, 3rd and 4th CrossAttnUpBlock2D
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples,
                                        encoder_hidden_states=encoder_hidden_states)
            else:  # 1st is UpBlock2D
                sample = upsample_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DiffDISPipeline(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, vae):
        super().__init__()
        self.vae = vae
        self.unet = unet

    def __call__(self,
                 input_image: torch.Tensor,
                 positive: torch.Tensor,
                 size: Tuple,
                 batch_size: int = 1,
                 show_progress_bar: bool = True,
                 ) -> torch.Tensor:
        # This is a wrapper to process batch_size images at a time
        single_rgb_dataset = TensorDataset(input_image)
        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=batch_size, shuffle=False)

        mask_pred_ls = []
        edge_pred_ls = []

        if show_progress_bar:
            iterable_bar = tqdm(single_rgb_loader, desc="Inference batches", leave=False)
        else:
            iterable_bar = single_rgb_loader

        for batch in iterable_bar:
            mask_pred, edge_pred = self.single_infer(input_rgb=batch[0], positive=positive, show_pbar=show_progress_bar)
            # Scale prediction to [0, 1]
            mask_pred = (mask_pred - torch.min(mask_pred)) / (torch.max(mask_pred) - torch.min(mask_pred))
            # Back to the original size, move results to CPU
            mask_pred_ls.append(torch.nn.functional.interpolate(mask_pred, size=size, mode=RESCALE_MODE).squeeze(1).cpu())
            edge_pred = (edge_pred - torch.min(edge_pred)) / (torch.max(edge_pred) - torch.min(edge_pred))
            edge_pred_ls.append(torch.nn.functional.interpolate(edge_pred, size=size, mode=RESCALE_MODE).squeeze(1).cpu())

        torch.cuda.empty_cache()  # clear vram cache for ensembling

        return torch.concat(mask_pred_ls, axis=0), torch.concat(edge_pred_ls, axis=0)

    def single_infer(self, input_rgb: torch.Tensor, positive: torch.Tensor, show_pbar: bool):
        device = input_rgb.device
        bsz = input_rgb.shape[0]
        wdtype = self.unet.conv_in.weight.dtype

        # Encode image 1:1 1/2 1/4 1/8 size
        rgb_latent = self.encode_RGB(input_rgb, device).to(wdtype)  # 1/8 Resolution with a channel nums of 4.
        # Latent for the mask and edge
        mask_edge_latent = torch.randn(rgb_latent.shape, device=device, dtype=wdtype).repeat(2, 1, 1, 1)
        rgb_latent = rgb_latent.repeat(2, 1, 1, 1)
        rgb_rsz2_lats = self.encode_RGB(resize(input_rgb, size=input_rgb.shape[-1]//2), device).to(wdtype).repeat(2, 1, 1, 1)
        rgb_rsz4_lats = self.encode_RGB(resize(input_rgb, size=input_rgb.shape[-1]//4), device).to(wdtype).repeat(2, 1, 1, 1)
        rgb_rsz8_lats = self.encode_RGB(resize(input_rgb, size=input_rgb.shape[-1]//8), device).to(wdtype).repeat(2, 1, 1, 1)

        # batched text embedding
        batch_text_embed = positive.repeat((bsz, 1, 1))  # [B, 2, 1024]

        # batch discriminative embedding
        discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=wdtype, device=device)
        BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)

        # The model works in 1 step, no need for scheduler
        self.unet.to(device)
        noise_pred = self.unet(
            torch.cat([rgb_latent, mask_edge_latent], dim=1),  # Input, order is important: [1, IN_CHANNELS, H, W] (8)
            torch.tensor([999, 999], device=device),           # Time steps, just the last one
            encoder_hidden_states=batch_text_embed.repeat(2, 1, 1),
            class_labels=BDE,
            rgb_token=[rgb_latent, rgb_rsz2_lats, rgb_rsz4_lats, rgb_rsz8_lats])  # -> [B, OUT_CHANNELS, h, w] (4)
        self.unet.cpu()
        # compute x_T -> x_0
        # mask_edge_latent = (mask_edge_latent - 0.9976672442 * noise_pred) * 14.64896838
        mask_edge_latent = (mask_edge_latent - noise_pred) * 14.64896838  # Almost the same

        return self.decode(mask_edge_latent, device)

    def encode_RGB(self, rgb_in: torch.Tensor, device: torch.device) -> torch.Tensor:
        # encode, ComfyUI returns the mean (deterministic)
        mean = self.vae.encode(rgb_in.movedim(1, -1)).to(device)
        # scale latent
        return mean * LATENT_SCALE_FACTOR

    def decode(self, mask_edge_latent: torch.Tensor, device: torch.device) -> torch.Tensor:
        # scale latents
        mask_edge_latent = mask_edge_latent / LATENT_SCALE_FACTOR
        # decode
        stacked = self.vae.decode(mask_edge_latent).movedim(-1, 1).to(device)
        # mean of output channels
        mask_stacked, edge_stacked = torch.chunk(stacked, 2, dim=0)
        mask_mean = mask_stacked.mean(dim=1, keepdim=True)
        mask_mean = torch.clip(mask_mean, -1.0, 1.0)
        mask_mean = (mask_mean + 1.0) / 2.0

        edge_mean = edge_stacked.mean(dim=1, keepdim=True)
        edge_mean = torch.clip(edge_mean, -1.0, 1.0)
        edge_mean = (edge_mean + 1.0) / 2.0

        return mask_mean, edge_mean
