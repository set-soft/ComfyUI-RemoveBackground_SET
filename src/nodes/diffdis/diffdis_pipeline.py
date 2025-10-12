from dataclasses import dataclass
import math
from typing import Dict, Union, Optional, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel
import torch.nn.functional as F

from ..diffusers.unet_2d_blocks import get_down_block, get_mid_block, get_up_block
from ..diffusers.modeling_outputs import BaseOutput


def resize(img, size):
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU())


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / half_dim

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(timesteps, self.num_channels)
        return t_emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, True)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class DiffDIS(nn.Module):
    def __init__(
        self,
        sample_size: int = 96,
        in_channels: int = 8,
        out_channels: int = 4,
        class_embed_type: str = 'projection',
        projection_class_embeddings_input_dim: int = 4,
        mid_extra_cross: bool = True,
        mode: str = 'DBIA',
        use_swci: bool = True,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.mid_extra_cross = mid_extra_cross
        self.mode = mode
        self.use_swci = use_swci
        self.class_embed_type = class_embed_type

        # block_out_channels
        boc0 = 320
        boc1 = boc0 * 2
        boc2 = boc1 * 2
        boc3 = boc2
        block_out_channels = [boc0, boc1, boc2, boc3]

        if self.use_swci:
            self.rgb_proj = nn.ModuleList([zero_module(nn.Conv2d(boc0, boc0, kernel_size=3, padding=1)),
                                           zero_module(nn.Conv2d(boc1, boc1, kernel_size=3, padding=1)),
                                           zero_module(nn.Conv2d(boc2, boc2, kernel_size=3, padding=1))])

            self.rgb_proj1 = nn.ModuleList([zero_module(nn.Conv2d(boc0, boc0, kernel_size=3, padding=1)),
                                            zero_module(nn.Conv2d(boc1, boc1, kernel_size=3, padding=1)),
                                            zero_module(nn.Conv2d(boc2, boc2, kernel_size=3, padding=1))])
            self.rgb_conv = nn.ModuleList([make_cbg(4, boc0), make_cbg(4, boc1), make_cbg(4, boc2)])

        attention_head_dim = num_attention_heads = [5, 10, 20, 20]

        # input
        self.conv_in = nn.Conv2d(in_channels, boc0, kernel_size=3, padding=1)

        # time
        time_embed_dim = boc0 * 4
        self.time_proj = Timesteps(boc0)
        timestep_input_dim = boc0
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        self.encoder_hid_proj = None

        # class embedding
        self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        # Not defined as addition_embed_type is None
        #    self.add_time_proj
        #    self.add_embedding
        self.time_embed_act = None

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        blocks_time_embed_dim = time_embed_dim

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
                transformer_layers_per_block=1,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resnet_groups=32,
                cross_attention_dim=1024,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                dual_cross_attention=False,
                use_linear_projection=True,
                only_cross_attention=False,
                upcast_attention=False,
                resnet_time_scale_shift="default",
                attention_type="default",
                resnet_skip_time_act=False,
                resnet_out_scale_factor=1.0,
                cross_attention_norm=None,
                attention_head_dim=attention_head_dim[i],
                dropout=0.0,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = get_mid_block(
            "UNetMidBlock2DCrossAttn",
            temb_channels=blocks_time_embed_dim,
            in_channels=block_out_channels[-1],
            resnet_eps=1e-5,
            resnet_act_fn="silu",
            resnet_groups=32,
            output_scale_factor=1,
            transformer_layers_per_block=1,
            num_attention_heads=num_attention_heads[-1],
            cross_attention_dim=1024,
            dual_cross_attention=False,
            use_linear_projection=True,
            mid_block_only_cross_attention=False,
            upcast_attention=False,
            resnet_time_scale_shift="default",
            attention_type="default",
            resnet_skip_time_act=False,
            cross_attention_norm=None,
            attention_head_dim=attention_head_dim[-1],
            dropout=0.0,
            mid_extra_cross=True,
            mode='DBIA',
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=3,
                transformer_layers_per_block=1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resnet_act_fn="silu",
                resolution_idx=i,
                resnet_groups=32,
                cross_attention_dim=1024,
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                use_linear_projection=True,
                only_cross_attention=False,
                upcast_attention=False,
                resnet_time_scale_shift="default",
                attention_type="default",
                resnet_skip_time_act=False,
                resnet_out_scale_factor=1.0,
                cross_attention_norm=None,
                attention_head_dim=attention_head_dim[i],
                dropout=0.0,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        rgb_token: Optional[Tuple[torch.Tensor]] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        The [`UNet2DConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond: (`torch.Tensor`, *optional*, defaults to `None`):
                Conditional embeddings for timestep. If provided, the embeddings will be summed with the samples passed
                through the `self.time_embedding` layer to obtain the timestep embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containing additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals: (`tuple` of `torch.Tensor`, *optional*):
                A tuple of tensors that if specified are added to the residuals of down unet blocks.
            mid_block_additional_residual: (`torch.Tensor`, *optional*):
                A tensor that if specified is added to the residual of the middle unet block.
            encoder_attention_mask (`torch.Tensor`):
                A cross-attention mask of shape `(batch, sequence_length)` is applied to `encoder_hidden_states`. If
                `True` the mask is kept, otherwise if `False` it is discarded. Mask will be converted into a bias,
                which adds large negative values to the attention scores corresponding to "discard" tokens.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttnProcessor`].
            added_cond_kwargs: (`dict`, *optional*):
                A kwargs dictionary containin additional embeddings that if specified are added to the embeddings that
                are passed along to the UNet blocks.
            down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added to UNet long skip connections from down blocks to up blocks for
                example from ControlNet side model(s)
            mid_block_additional_residual (`torch.Tensor`, *optional*):
                additional residual to be added to UNet mid block output, for example from ControlNet side model
            down_intrablock_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                additional residuals to be added within UNet down blocks, for example from T2I-Adapter side model(s)

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d_condition.UNet2DConditionOutput`] is returned, otherwise
                a `tuple` is returned where the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                # Forward upsample size to force interpolation output size.
                forward_upsample_size = True
                break

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        # if self.config.center_input_sample:
        #     sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            emb = emb + class_emb

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # sample = self.fft(sample)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # 3. down
        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            # lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        # else:
            # lora_scale = 1.0

        # if USE_PEFT_BACKEND:
        #     # weight the lora layers by setting `lora_scale` for each PEFT layer
        #     scale_lora_layers(self, lora_scale)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        # using new arg down_intrablock_additional_residuals for T2I-Adapters, to distinguish from controlnets
        is_adapter = down_intrablock_additional_residuals is not None
        # maintain backward compatibility for legacy usage, where
        #       T2I-Adapter and ControlNet both use down_block_additional_residuals arg
        #       but can only use one or the other
        if not is_adapter and mid_block_additional_residual is None and down_block_additional_residuals is not None:
            assert False
            down_intrablock_additional_residuals = down_block_additional_residuals
            is_adapter = True

        down_block_res_samples = (sample,)
        for j, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
                if is_adapter and len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

            if self.use_swci and j < 3:

                rgb_latents = self.rgb_conv[j](rgb_token[j+1])
                rgb_latents_zero = self.rgb_proj[j](rgb_latents)
                rgb_latents_zero1 = self.rgb_proj1[j](rgb_latents)

                down_block_res_samples = list(down_block_res_samples)
                down_block_res_samples[-1] = (((down_block_res_samples[-1] * rgb_latents_zero) + rgb_latents_zero1) +
                                              down_block_res_samples[-1])
                down_block_res_samples = tuple(down_block_res_samples)

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

            # To support T2I-Adapter-XL
            if (
                is_adapter
                and len(down_intrablock_additional_residuals) > 0
                and sample.shape == down_intrablock_additional_residuals[0].shape
            ):
                sample += down_intrablock_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # if USE_PEFT_BACKEND:
        #    # remove `lora_scale` from each PEFT layer
        #    unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)
        else:
            return UNet2DConditionOutput(sample=sample)


class DiffDISPipeline(torch.nn.Module):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    mask_latent_scale_factor = 0.18215
    weight_dtype = torch.float32

    def __init__(self, unet: UNet2DConditionModel, vae):
        super().__init__()
        self.vae = vae
        self.unet = unet

    def __call__(self,
                 input_image: torch.Tensor,
                 positive: torch.Tensor,
                 batch_size: int = 0,
                 show_progress_bar: bool = True,
                 ) -> torch.Tensor:

        single_rgb_dataset = TensorDataset(input_image)
        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=batch_size, shuffle=False)

        mask_pred_ls = []
        edge_pred_ls = []

        if show_progress_bar:
            iterable_bar = tqdm(single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False)
        else:
            iterable_bar = single_rgb_loader

        for batch in iterable_bar:
            mask_pred, edge_pred = self.single_infer(
                input_rgb=batch[0],
                positive=positive,
                show_pbar=show_progress_bar
            )
            mask_pred_ls.append(mask_pred.detach().clone())
            edge_pred_ls.append(edge_pred.detach().clone())

        mask_preds = torch.concat(mask_pred_ls, axis=0)
        edge_preds = torch.concat(edge_pred_ls, axis=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Post processing -----------------
        # scale prediction to [0, 1]
        mask_preds = (mask_preds - torch.min(mask_preds)) / (torch.max(mask_preds) - torch.min(mask_preds))
        edge_preds = (edge_preds - torch.min(edge_preds)) / (torch.max(edge_preds) - torch.min(edge_preds))

        return mask_preds, edge_preds

    def single_infer(self, input_rgb: torch.Tensor,
                     positive: torch.Tensor,
                     show_pbar: bool):

        device = input_rgb.device
        bsz = input_rgb.shape[0]
        wdtype = self.weight_dtype

        # Encode image 1:1 1/2 1/4 1/8 size
        rgb_latent = self.encode_RGB(input_rgb).to(wdtype)  # 1/8 Resolution with a channel nums of 4.
        # Latent for the mask and edge
        mask_edge_latent = torch.randn(rgb_latent.shape, device=device, dtype=wdtype).repeat(2, 1, 1, 1)
        rgb_latent = rgb_latent.repeat(2, 1, 1, 1)
        rgb_resized2_latents = self.encode_RGB(resize(input_rgb, size=input_rgb.shape[-1]//2)).to(wdtype).repeat(2, 1, 1, 1)
        rgb_resized4_latents = self.encode_RGB(resize(input_rgb, size=input_rgb.shape[-1]//4)).to(wdtype).repeat(2, 1, 1, 1)
        rgb_resized8_latents = self.encode_RGB(resize(input_rgb, size=input_rgb.shape[-1]//8)).to(wdtype).repeat(2, 1, 1, 1)

        # batched text embedding
        batch_text_embed = positive.repeat((bsz, 1, 1))  # [B, 2, 1024]

        # batch discriminative embedding
        discriminative_label = torch.tensor([[0, 1], [1, 0]], dtype=wdtype, device=device)
        BDE = torch.cat([torch.sin(discriminative_label), torch.cos(discriminative_label)], dim=-1).repeat_interleave(bsz, 0)

        # The model works in 1 step, no need for scheduler
        self.unet.to(device)
        unet_input = torch.cat([rgb_latent, mask_edge_latent], dim=1)  # this order is important: [1,8,H,W]
        t = torch.tensor([999, 999], device=device)
        noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_text_embed.repeat(2, 1, 1),
                               class_labels=BDE, rgb_token=[rgb_latent, rgb_resized2_latents,
                               rgb_resized4_latents, rgb_resized8_latents]).sample  # [B, 4, h, w]
        # compute x_T -> x_0
        # mask_edge_latent = (mask_edge_latent - 0.9976672442 * noise_pred) * 14.64896838
        mask_edge_latent = (mask_edge_latent - noise_pred) * 14.64896838  # Almost the same
        self.unet.cpu()

        mask, edge = self.decode(mask_edge_latent)

        mask = torch.clip(mask, -1.0, 1.0)
        mask = (mask + 1.0) / 2.0

        edge = torch.clip(edge, -1.0, 1.0)
        edge = (edge + 1.0) / 2.0

        return mask, edge

    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        # encode, ComfyUI returns the mean (deterministic)
        mean = self.vae.encode(rgb_in.movedim(1, -1)).cuda()
        # scale latent
        return mean * self.rgb_latent_scale_factor

    def decode(self, mask_edge_latent: torch.Tensor) -> torch.Tensor:
        # scale latents
        mask_edge_latent = mask_edge_latent / self.mask_latent_scale_factor
        # decode
        stacked = self.vae.decode(mask_edge_latent).movedim(-1, 1).cuda()
        # mean of output channels
        mask_stacked, edge_stacked = torch.chunk(stacked, 2, dim=0)
        mask_mean = mask_stacked.mean(dim=1, keepdim=True)
        edge_mean = edge_stacked.mean(dim=1, keepdim=True)

        return mask_mean, edge_mean
