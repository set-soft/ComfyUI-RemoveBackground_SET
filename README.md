# ComfyUI Remove Background nodes (SET)

This repository provides a set of custom nodes for ComfyUI focused on background removal and or replacement.

![Remove Background](https://raw.githubusercontent.com/set-soft/ComfyUI-RemoveBackground_SET/main/doc/Banner.jpg)

## &#x2699;&#xFE0F; Main features

&#x2705; No bizarre extra dependencies, we use the same modules as ComfyUI

&#x2705; Warnings and errors visible in the browser, configurable debug information in the console

&#x2705; Support for BEN1/2, BiRefNet, BRIA 1.4/2, Depth Anything V2, DiffDIS, InSPyReNet, MODNet, MVANet, PDFNet, U-2-Net, IS-Net

&#x2705; Automatic model download (Only the SD Turbo VAE might be needed for DiffDIS)


## &#x0001F4DC; Table of Contents

- &#x0001F680; [Installation](#-installation)
- &#x0001F4E6; [Dependencies](#-dependencies)
- &#x0001F5BC;&#xFE0F; [Examples](#&#xFE0F;-examples)
   - [Simple](#simple) (01_Simple, 01_Change_Background)
   - [More advanced](#more-advanced) (02_Full_example, 03_Web_page_examples, 04_Advanced)
   - [Video](#video) (05_Video, 05_Video_Advanced)
   - [Model specific](#model-specific) (01_PDFNet_simple, 06_PDFNet_external_map, 01_Simple_DiffDIS)
   - [Comparison](#comparison) (07_PDFNet_vs_BiRefNet, 09_Compare_Models)
- &#x2728; [Nodes](#-nodes)
   - [Loaders](#loaders)
      - [Load RemBG model by file](#load-rembg-model-by-file)
      - [Load XXXXXX model by name](#load-xxxxxx-model-by-name)
   - [Processing nodes](#processing-nodes)
      - [Remove background](#remove-background)
      - [Remove background (full)](#remove-background-full)
      - [Get background mask](#get-background-mask)
      - [Get background mask low level](#get-background-mask-low-level)
   - [Other nodes](#other-nodes)
      - [Load Depth Anything by name](#load-depth-anything-by-name)
      - [Depth Anything V2](#depth-anything-v2)
- &#x0001F4DD; [Usage Notes](#-usage-notes)
- &#x0001F4DC; [Project History](#-project-history)
- &#x2696;&#xFE0F; [License](#&#xFE0F;-license)
- &#x0001F64F; [Attributions](#-attributions)

## &#x2728; Nodes

### Loaders

The loaders are used to load a background removal model. We have a general loader that will look for models in the `ComfyUI/models/RemBG` folder.
You can reconfigure this path using the `rembg` key in the `extra_model_paths.yaml` file of ComfyUI.

In addition we have automatic downloaders for each supported model family.

### Load RemBG model by file
   - **Display Name:** `Load RemBG model by file`
   - **Internal Name:** `LoadRembgByBiRefNetModel_SET`
   - **Category:** `RemBG_SET/Load`
   - **Description:** Loads a model from the `ComfyUI/models/RemBG` folder, you can connect its output to any of the processing nodes
   - **Purpose:** Used for models that you already downloaded, or pehaps you trained.
   - **Inputs:**
     - `model` (`FILENAME`): The name of the model in the `ComfyUI/models/RemBG` folder, use `R` to refresh the list
     - `device` (`DEVICE`): The device where the model will be executed. Using `AUTO` you'll use the default ComfyUI target (i.e. your GPU)
     - `dtype` (`DTYPE_OPS`): Used to select the data type using during inference. `AUTO` means we will use the same data type as the model weights loaded from disk. Most of the models performs quite well on 16 bits floating point. You can force 16 bits to save VRAM, or even force to convert 16 bits values to 32 bits.
     - `vae` (`VAE`, optional): Only needed for DiffDIS, you have to connect a "Load VAE" node here. The model needs the SD Turbo VAE, please look in the examples.
     - `positive` (`CONDITIONING`, optional): Experimental and used only for DiffDIS. In practice you should leave it unconnected, the model was trained with an empty conditioning text.
   - **Output:**
     - `model` (`SET_REMBG`): The loaded model, ready to be connected to a processing node

### Load XXXXXX model by name
   - **Display Name:** `Load XXXXXX model by name`
   - **Internal Name:** `AutoDownloadXXXXXXModel_SET`
   - **Category:** `RemBG_SET/Load`
   - **Description:** Load a model of the XXXXXX family, if the model isn't on disk this is automatically downloaded. XXXXXX is one of the supported families (i.e. 'BiRefNet', 'MVANet/BEN', 'InSPyReNet', 'U-2-Net', 'IS-Net', 'MODNet', 'PDFNet', 'DiffDIS')
   - **Purpose:** Download from internet and load to memory a model for bacground removal. The names are descriptive and says how big is the file.
   - **Inputs:**
     - `model` (`FILENAME`): The descriptive name of the model
     - `device` (`DEVICE`): The device where the model will be executed. Using `AUTO` you'll use the default ComfyUI target (i.e. your GPU)
     - `dtype` (`DTYPE_OPS`): Used to select the data type using during inference. `AUTO` means we will use the same data type as the model weights loaded from disk. Most of the models performs quite well on 16 bits floating point. You can force 16 bits to save VRAM, or even force to convert 16 bits values to 32 bits.
     - `vae` (`VAE`, only for DiffDIS): You have to connect a "Load VAE" node here. The model needs the SD Turbo VAE, please look in the examples.
     - `positive` (`CONDITIONING`, only for DiffDIS): Experimental. In practice you should leave it unconnected, the model was trained with an empty conditioning text.
   - **Output:**
     - `model` (`SET_REMBG`): The loaded model, ready to be connected to a processing node
     - `train_w` (`INT`): Width of the images used during training. You should use this size for optimal results. Note that most models accepts any size multiple of 32. The `General 2K Lite` BiRefNet model was trained with some flexibility in the size. The DiffDIS is more restrictive. Only for manual pre-processing.
     - `train_h` (`INT`): Height of the images used during training. You should use this size for optimal results. Note that most models accepts any size multiple of 32. The `General 2K Lite` BiRefNet model was trained with some flexibility in the size. The DiffDIS is more restrictive. Only for manual pre-processing.
     - `norm_params` (`NORM_PARAMS`): Normalization parameters for the input images. This is needed only for advanced use when you want to manually pre-process the images. The `Arbitrary Normalize` node from [Image Misc](https://github.com/set-soft/ComfyUI-ImageMisc) can use these parameters to apply the correct normalization.


### Processing nodes

These nodes applies the loaded model to estimate the foreground object.
For normal use the simplest node is `Remove background`, it will generate an RGBA image with transparent background, or replace it using a provided image.

This simple node doesn't allow to change much options, so you could also want to use `Remove background (full)`. But note this node will also consume more RAM.

What the models do is to generate a map where each pixel represents the estimated probability that it belongs to the foreground. This is mask that you can apply to remove the background.
The `Get background mask` node is oriented to just get this mask, no background removal or replacement.

If you want to play at the lowest level use the `Get background mask low level`. This node doesn't pre-process the input image and returns the mask without extra post-processing.


### Remove background
   - **Display Name:** `Remove background`
   - **Internal Name:** `RembgByBiRefNet_SET`
   - **Category:** `RemBG_SET/Basic`
   - **Description:** Removes or replaces the background from the input image.
   - **Inputs:**
     - `model` (`SET_REMBG`): The model to use
     - `images` (`IMAGE`): One or more images to process, will be scaled to a size that is good for the model
     - `batch_size` (`INT`): How many images will be processed at once. Useful for videos, depending on the model and the GPU you have this can make things much faster. Consumes more VRAM.
     - `depths` (`MASK` optional): Can be used for PDFNet to provide externally computed depth maps
     - `background` (`IMAGE` optional): Image to use as background, will be scaled to the size of `images`. If you don't provide an image the output will be an RGBA image with transparency. Note that this can be 1 or more images. If `images` is a video this input can be another video of the same number of frames.
     - `out_dtype` (`AUTO`, `float32`, `float16`): Which data type will be used for the output image. Using `AUTO` is recommended. Can be used to save RAM when processing long videos, use `float16`.
   - **Output:**
     - `images` (`IMAGE`): The images with the background removed (transparent) or replaced by the background image


### Remove background (full)
   - **Display Name:** `Remove background (full)`
   - **Internal Name:** `RembgByBiRefNetAdvanced_SET`
   - **Category:** `RemBG_SET/Advanced`
   - **Description:** Removes or replaces the background from the input image. Gives more options and also generates masks and other stuff.
   - **Inputs:**
     - `model` (`SET_REMBG`): The model to use
     - `images` (`IMAGE`): One or more images to process, will be scaled to a size that is good for the model
     - `width` (`INT`): The width to scale the image before applying the model. Should be supported by the model. Usually is the `train_w` from `Load XXXXXX model by name`
     - `height` (`INT`): The height to scale the image before applying the model. Should be supported by the model. Usually is the `train_h` from `Load XXXXXX model by name`
     - `upscale_method` (`area`, `bicubic`, `nearest-exact`, `bilinear`, `lanczos`): Which algorithm will be used to scale the image to and from the model size. Usually `bicubic` is a good choice.
     - `blur_size` (`INT`): Diameter for the coarse gaussian blur used for the [Approximate Fast Foreground Colour Estimation](https://github.com/Photoroom/fast-foreground-estimation).
     - `blur_size_two` (`INT`): Diameter for the fine gaussian blur (see `blur_size`)
     - `fill_color` (`BOOLEAN`): When enabled and no background is provided we fill the background using a color.
     - `color` (`STRING`): Color to use when filling the bacground. You can sepcify it in multiple ways, even by name, [more here](https://pillow.readthedocs.io/en/stable/reference/ImageColor.html)
     - `mask_threshold` (`FLOAT`): Most models generates masks that contain a value from 0 to 1, but can be any value in between. Matte models can estimate transparency using it. If you need to make the mask 0 or 1, but nothing in between, you can provide a threshold here. Values above it will become 1 and the rest 0.
     - `batch_size` (`INT`): How many images will be processed at once. Useful for videos, depending on the model and the GPU you have this can make things much faster. Consumes more VRAM.
     - `depths` (`MASK` optional): Can be used for PDFNet to provide externally computed depth maps. This is the map generated by `Depth Anything V2`
     - `background` (`IMAGE` optional): Image to use as background, will be scaled to the size of `images`. If you don't provide an image the output will be an RGBA image with transparency. Note that this can be 1 or more images. If `images` is a video this input can be another video of the same number of frames.
     - `out_dtype` (`AUTO`, `float32`, `float16`): Which data type will be used for the output image. Using `AUTO` is recommended. Can be used to save RAM when processing long videos, use `float16`.
   - **Output:**
     - `images` (`IMAGE`): The images with the background removed (transparent) or replaced by the background image
     - `masks` (`MASK`): The estimated masks, where a higher value means the model estimates it belongs to the foreground with more confidence.
     - `depths` (`MASK`): The estimated depth map. Either from the `depths` input or computed. Note this applies only to PDFNet. This is the map generated by `Depth Anything V2`
     - `edges` (`MASK`): The estimated edges. This is only generated by the DiffDIS model.


### Get background mask
   - **Display Name:** `Get background mask`
   - **Internal Name:** `GetMaskByBiRefNet_SET`
   - **Category:** `RemBG_SET/Basic`
   - **Description:** Computes the foreground mask. It normalizes the input images, scales them to the model size, computes the masks and then scales the masks to the image size. No background removal/replacement is done.
   - **Inputs:**
     - `model` (`SET_REMBG`): The model to use
     - `images` (`IMAGE`): One or more images to process, will be scaled to a size that is good for the model
     - `width` (`INT`): The width to scale the image before applying the model. Should be supported by the model. Usually is the `train_w` from `Load XXXXXX model by name`
     - `height` (`INT`): The height to scale the image before applying the model. Should be supported by the model. Usually is the `train_h` from `Load XXXXXX model by name`
     - `upscale_method` (`area`, `bicubic`, `nearest-exact`, `bilinear`, `lanczos`): Which algorithm will be used to scale the image to and from the model size. Usually `bicubic` is a good choice.
     - `mask_threshold` (`FLOAT`): Most models generates masks that contain a value from 0 to 1, but can be any value in between. Matte models can estimate transparency using it. If you need to make the mask 0 or 1, but nothing in between, you can provide a threshold here. Values above it will become 1 and the rest 0.
     - `batch_size` (`INT`): How many images will be processed at once. Useful for videos, depending on the model and the GPU you have this can make things much faster. Consumes more VRAM.
     - `depths` (`MASK` optional): Can be used for PDFNet to provide externally computed depth maps. This is the map generated by `Depth Anything V2`
     - `out_dtype` (`AUTO`, `float32`, `float16`): Which data type will be used for the output image. Using `AUTO` is recommended. Can be used to save RAM when processing long videos, use `float16`.
   - **Output:**
     - `masks` (`MASK`): The estimated masks, where a higher value means the model estimates it belongs to the foreground with more confidence.
     - `depths` (`MASK`): The estimated depth map. Either from the `depths` input or computed. Note this applies only to PDFNet. This is the map generated by `Depth Anything V2`
     - `edges` (`MASK`): The estimated edges. This is only generated by the DiffDIS model.


### Get background mask low level
   - **Display Name:** `Get background mask low level`
   - **Internal Name:** `GetMaskLowByBiRefNet_SET`
   - **Category:** `RemBG_SET/Advanced`
   - **Description:** Computes the foreground mask. No pre or post processing is applied, you must do it outside the node.
   - **Inputs:**
     - `model` (`SET_REMBG`): The model to use
     - `images` (`IMAGE`): One or more images to process, they must be normalized to a range that is good for the model. Their size must be similar to the size used to train the model.
     - `batch_size` (`INT`): How many images will be processed at once. Useful for videos, depending on the model and the GPU you have this can make things much faster. Consumes more VRAM.
     - `depths` (`MASK` optional): Can be used for PDFNet to provide externally computed depth maps. This is the map generated by `Depth Anything V2`
     - `out_dtype` (`AUTO`, `float32`, `float16`): Which data type will be used for the output image. Using `AUTO` is recommended. Can be used to save RAM when processing long videos, use `float16`.
   - **Output:**
     - `masks` (`MASK`): The estimated masks, where a higher value means the model estimates it belongs to the foreground with more confidence.
     - `depths` (`MASK`): The estimated depth map. Either from the `depths` input or computed. Note this applies only to PDFNet. This is the map generated by `Depth Anything V2`
     - `edges` (`MASK`): The estimated edges. This is only generated by the DiffDIS model.


### Other nodes

The PDFNet model is a special case, instead of just using the image it also uses an estimation of the depth of the image performed using
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2). In order to allow automatic computation of the depth maps I took
[Kijai](https://github.com/kijai) [nodes](https://github.com/kijai/ComfyUI-DepthAnythingV2) and adapted them to this use.


### Load Depth Anything by name
   - **Display Name:** `Load Depth Anything by name`
   - **Internal Name:** `DownloadAndLoadDepthAnythingV2Model_SET`
   - **Category:** `RemBG_SET/Load`
   - **Description:** Downloads and loads to memory one of the Depth Anything V2 models.
   - **Inputs:**
     - `model` (`STRING`): The name of the model to use. Small, Base and Large are available in 16 and 32 bits. The 16 bits version works quite well. PDFNet was trained using the Base version.
   - **Output:**
     - `da_v2_model` (`DAMODEL`): The model ready to be used.


### Depth Anything V2
   - **Display Name:** `Depth Anything V2`
   - **Internal Name:** `DepthAnything_V2_SET`
   - **Category:** `RemBG_SET/Advanced`
   - **Description:** Computes an estimated depth map of the image, larger values means the pixel is closer to the camera
   - **Inputs:**
     - `da_model` (`DAMODEL`): The model from the `Load Depth Anything by name` node.
     - `images` (`IMAGE`): One or more images to process, will be normalized and scaled.
     - `batch_size` (`INT`): How many images will be processed at once.
   - **Output:**
     - `depths` (`MASK`): The depth map
     - `depth_imgs` (`IMAGE`): The same map in a format compatible with nodes that needs an image. The three channels (R, G, B) are the same buffer, shared with the mask.


## &#x0001F680; Installation

You can install the nodes from the ComfyUI nodes manager, the name is *Remove Background (SET)* (remove-background), or just do it manually:

1.  Clone this repository into your `ComfyUI/custom_nodes/` directory:
    ```bash
    cd ComfyUI/custom_nodes/
    git clone https://github.com/set-soft/ComfyUI-RemoveBackground_SET ComfyUI-RemoveBackground_SET
    ```
2.  Install dependencies: `pip install -r ComfyUI/custom_nodes/ComfyUI-RemoveBackground_SET/requirements.txt`
3.  Restart ComfyUI.

The nodes should then appear under the "RemBG_SET" category in the "Add Node" menu.


## &#x0001F4E6; Dependencies

- SeCoNoHe (seconohe): This is just some functionality I wrote shared by my nodes, only depends on ComfyUI.
- PyTorch: Installed by ComfyUI
- einops: Installed by ComfyUI
- kornia: Installed by ComfyUI
- safetensors: Installed by ComfyUI
- Requests (optional): Usually an indirect ComfyUI dependency. If installed it will be used for downloads, it should be more robust than then built-in `urllib`, used as fallback.
- Colorama (optional): Might help to get colored log messages on some terminals. We use ANSI escape sequences when it isn't installed.


## &#x0001F5BC;&#xFE0F; Examples

Once installed the examples are available in the ComfyUI workflow templates, in the *remove-background* section (or ComfyUI-RemoveBackground_SET).

### Simple

These examples shows how to remove the background, obtaining an image with transparency, or replacing it with an image.
Note that RGBA images, the ones with transparency, aren't supported by all nodes.
The correct way to handle them is to have the image and a mask, but using RGBA is what most background removal tools do.

- [01_Simple](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/01_Simple.json): Basic use to get an RGBA image
- [01_Change_Background](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/01_Change_Background.json): Basic example showing how to replace the background of an image

### More advanced

These examples show how to have more control over the process.

- [02_Full_example](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/02_Full_example.json): Shows how to use the full node to get an RGBA image. Needs [Image Misc](https://github.com/set-soft/ComfyUI-ImageMisc) to download the example image.
- [03_Web_page_examples](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/03_Web_page_examples.json): Allows comparing the result with the original image. Downloads the BiRefNet exmamples. Needs [Image Misc](https://github.com/set-soft/ComfyUI-ImageMisc) to download the example images and [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) to compare the images.
- [04_Advanced](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/04_Advanced.json): Shows how to do custom pre and post processing, including filling with a color, background image replacement and object highlight. Needs [Image Misc](https://github.com/set-soft/ComfyUI-ImageMisc) and [rgthree-comfy](https://github.com/rgthree/rgthree-comfy) to compare the images.

### Video

Examples for video processing, using ComfyUI video nodes and advanced [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) nodes.

- [05_Video](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/05_Video.json): Simple video workflow to replace the background of a video using a still image. Uses the Comfy-Core nodes.
- [![05_Video_Advanced](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/05_Video_Advanced.jpg)](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/05_Video_Advanced.json): Video workflow to replace the background of a video using another video. Uses the [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) which allows resize, skip frames, limit frames, etc.

### Model specific

Examples related to particular models. PDFNet uses a depth map and DiffDIS is a diffusion model repurposed for DIS.

- [01_PDFNet_simple](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/01_PDFNet_simple.json): Shows how to use the PDFNet model and the automatically computed maps.
- [06_PDFNet_external_map](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/06_PDFNet_external_map.json): Shows how to use the PDFNet model and the externally computed maps.
- [01_Simple_DiffDIS](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/01_Simple_DiffDIS.json): Basic use to get an RGBA image using DiffDIS model

### Comparison

Example workflows showing how to compare the models.

- [![07_PDFNet_vs_BiRefNet](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/07_PDFNet_vs_BiRefNet.jpg)](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/07_PDFNet_vs_BiRefNet.json): Example to compare two models, in this case PDFNet vs BiRefNet
- [![09_Compare_Models](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/09_Compare_Models.jpg)](https://raw.githubusercontent.com/set-soft/RemoveBackground_SET/refs/heads/main/example_workflows/09_Compare_Models.json): Compares 10 models and generates an images showing the output from the 10 models. Needs [Image Misc](https://github.com/set-soft/ComfyUI-ImageMisc) to compose the final image.


## &#x0001F4DD; Usage Notes

- **Logging:** &#x0001F50A; The nodes use Python's `logging` module. Debug messages can be helpful for understanding the transformations being applied.
  You can control log verbosity through ComfyUI's startup arguments (e.g., `--preview-method auto --verbose DEBUG` for more detailed ComfyUI logs
  which might also affect custom node loggers if they are configured to inherit levels). The logger name used is "RemoveBackground_SET".
  You can force debugging level for these nodes defining the `REMOVEBACKGROUND_SET_NODES_DEBUG` environment variable to `1` or `2`.


## &#x0001F4DC; Project History

- 1.0.0 2025-10-??: Initial release


## &#x2696;&#xFE0F; License

[GPL-3.0](LICENSE)

## &#x0001F64F; Attributions

- Good part of the initial code and this README was generated using Gemini 2.5 Pro.
- I took various ideas from [ComfyUI_BiRefNet_ll](https://github.com/lldacing/ComfyUI_BiRefNet_ll)
- These nodes contains the inference code for the models:
  - [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet): Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, Nicu Sebe
  - [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2): Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao (HKU/TikTok)
  - [DiffDIS](https://github.com/qianyu-dlut/DiffDIS): Qian Yu, Peng-Tao Jiang, Hao Zhang, Jinwei Chen, Bo Li, Lihe Zhang, Huchuan Lu
  - [Diffusers](https://huggingface.co/docs/diffusers/index): The HuggingFace Team
  - [DINO](https://github.com/facebookresearch/dinov2): Meta AI Research
  - [InSPyReNet](https://github.com/plemeri/InSPyReNet): Taehun Kim, Kunhee Kim, Joonyeong Lee, Dongmin Cha, Jiho Lee, Daijin Kim
  - [MODNet](https://github.com/ZHKKKe/MODNet): Zhanghan Ke, Jiayu Sun, Kaican Li, Qiong Yan, Rynson W.H. Lau
  - [MVANet](https://github.com/qianyu-dlut/MVANet/): Qian Yu, Xiaoqi Zhao, Youwei Pang, Lihe Zhang, Huchuan Lu
    - [BEN](https://huggingface.co/PramaLLC/BEN2): Maxwell Meyer and Jack Spruyt
  - [PDFNet](https://github.com/Tennine2077/PDFNet): Xianjie Liu, Keren Fu, Qijun Zhao
  - [Swin](https://github.com/microsoft/Swin-Transformer): Ze Liu, Yutong Lin, Yixuan Wei
  - [U-2-Net](https://github.com/xuebinqin/U-2-Net): Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane and Martin Jagersand
    - [IS-Net](https://github.com/xuebinqin/DIS): Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, Luc Van Gool
- Code for Depth Anything v2 by [Kijai](https://github.com/kijai) (Jukka Seppänen)
- All working together by Salvador E. Tropea
