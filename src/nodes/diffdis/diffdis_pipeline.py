from typing import Dict
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from diffusers import UNet2DConditionModel
import torch.nn.functional as F


def resize(img, size):
    return F.interpolate(img, size=size, mode='bilinear', align_corners=False)


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
