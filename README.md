[中文文档](README_CN.md)

Support the use of new and old versions of BiRefNet models

## Preview
![save api extended](doc/base.png)
![save api extended](doc/video.gif)

## Install

- Manual
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_BiRefNet_ll.git
    cd ComfyUI_BiRefNet_ll
    pip install -r requirements.txt
    # restart ComfyUI
```
- Via ComfyUI Manager
    

## Models

### The available newest models are:

- General: A pre-trained model for general use cases.
- General-HR: A pre-trained model for general use cases which shows great performance on higher resolution images (2048x2048).
- General-Lite: A light pre-trained model for general use cases.
- General-Lite-2K: A light pre-trained model for general use cases in high resolution (2560x1440).
- General-dynamic: A pre-trained model for dynamic resolution, trained with images in range from 256x256 to 2304x2304.
- General-reso_512: A pre-trained model for faster and more accurate lower resolution, trained with images in 512x512.
- General-legacy: A pre-trained model for general use trained on DIS5K-TR,DIS-TEs, DUTS-TR_TE,HRSOD-TR_TE,UHRSD-TR_TE, HRS10K-TR_TE (w/o portrait seg data).
- Portrait: A pre-trained model for human portraits.
- Matting: A pre-trained model for general trimap-free matting use.
- Matting-HR: A pre-trained model for general matting use which shows great matting performance on higher resolution images (2048x2048).
- Matting-Lite: A light pre-trained model for general trimap-free matting use.
- DIS: A pre-trained model for dichotomous image segmentation (DIS).
- HRSOD: A pre-trained model for high-resolution salient object detection (HRSOD).
- COD: A pre-trained model for concealed object detection (COD).
- DIS-TR_TEs: A pre-trained model with massive dataset.

Model files go here (when use AutoDownloadBiRefNetModel automatically downloaded if the folder is not present during first run): `${comfyui_rootpath}/models/BiRefNet`.  

If necessary, they can be downloaded from:
- [General](https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General.safetensors`
- [General-HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General-HR.safetensors`
- [General-Lite](https://huggingface.co/ZhengPeng7/BiRefNet_T/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General-Lite.safetensors`
- [General-Lite-2K](https://huggingface.co/ZhengPeng7/BiRefNet_lite-2K/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General-Lite-2K.safetensors`
- [General-dynamic](https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General-dynamic.safetensors`
- [General-legacy](https://huggingface.co/ZhengPeng7/BiRefNet-legacy/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General-legacy.safetensors`
- [General-reso_512](https://huggingface.co/ZhengPeng7/BiRefNet_512x512/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `General-reso_512.safetensors`
- [Portrait](https://huggingface.co/ZhengPeng7/BiRefNet-portrait/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `Portrait.safetensors`
- [Matting](https://huggingface.co/ZhengPeng7/BiRefNet-matting/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `Matting.safetensors`
- [Matting-HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `Matting-HR.safetensors`
- [Matting-Lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite-matting/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `Matting-Lite.safetensors`
- [DIS](https://huggingface.co/ZhengPeng7/BiRefNet-DIS5K/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `DIS.safetensors`
- [HRSOD](https://huggingface.co/ZhengPeng7/BiRefNet-HRSOD/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `HRSOD.safetensors`
- [COD](https://huggingface.co/ZhengPeng7/BiRefNet-COD/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `COD.safetensors`
- [DIS-TR_TEs](https://huggingface.co/ZhengPeng7/BiRefNet-DIS5K-TR_TEs/resolve/main/model.safetensors) ➔ `model.safetensors` must be renamed `DIS-TR_TEs.safetensors`

Some models on GitHub: 
[BiRefNet Releases](https://github.com/ZhengPeng7/BiRefNet/releases)

### Old models:
- [BiRefNet-DIS_ep580.pth](https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-DIS_ep580.pth)
- [BiRefNet-ep480.pth](https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-ep480.pth)

## Weight Models (Optional)
- [swin_large_patch4_window12_384_22kto1k.pth](https://huggingface.co/ViperYX/BiRefNet/resolve/main/swin_large_patch4_window12_384_22kto1k.pth)(not General-Lite, General-Lite-2K and Matting-Lite model)
- [swin_tiny_patch4_window7_224_22kto1k_finetune.pth](https://drive.google.com/drive/folders/1cmce_emsS8A5ha5XT2c_CZiJzlLM81ms)(just General-Lite, General-Lite-2K and Matting-Lite model)


## Nodes
- AutoDownloadBiRefNetModel
  - Automatically download the model into `${comfyui_rootpath}/models/BiRefNet`, do not support weight model
- LoadRembgByBiRefNetModel
  - Can select model from `${comfyui_rootpath}/models/BiRefNet` or the path of `birefnet` configured in the extra YAML file
  - You can download latest models from [BiRefNet Releases](https://github.com/ZhengPeng7/BiRefNet/releases) or old models [BiRefNet-DIS_ep580.pth](https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-DIS_ep580.pth) and [BiRefNet-ep480.pth](https://huggingface.co/ViperYX/BiRefNet/resolve/main/BiRefNet-ep480.pth)
- RembgByBiRefNet
  - Output transparent foreground image and mask
- RembgByBiRefNetAdvanced
  - Output foreground image and mask, provide some fine-tuning parameters
- GetMaskByBiRefNet
  - Only output mask
- BlurFusionForegroundEstimation
  - Use the [fast-foreground-estimation method](https://github.com/Photoroom/fast-foreground-estimation) to estimate the foreground image

## Thanks

[ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet)

[dimitribarbot/sd-webui-birefnet](https://github.com/dimitribarbot/sd-webui-birefnet)

