# ComfyUI Remove Background nodes (SET)

This repository provides a set of custom nodes for ComfyUI focused on background removal and or replacement.

![Remove Background](https://raw.githubusercontent.com/set-soft/ComfyUI-RemoveBackground_SET/main/doc/Banner.jpg)

## &#x2699;&#xFE0F; Main features

&#x2705; No bizarre extra dependencies, we use the same modules as ComfyUI

&#x2705; Warnings and errors visible in the browser, configurable debug information in the console


## &#x0001F4DC; Table of Contents

- &#x0001F680; [Installation](#-installation)
- &#x0001F4E6; [Dependencies](#-dependencies)
- &#x0001F5BC;&#xFE0F; [Examples](#&#xFE0F;-examples)
- &#x2728; [Nodes](#-extra-nodes)
- &#x0001F4DD; [Usage Notes](#-usage-notes)
- &#x0001F4DC; [Project History](#-project-history)
- &#x2696;&#xFE0F; [License](#&#xFE0F;-license)
- &#x0001F64F; [Attributions](#-attributions)

## &#x2728; Nodes

The nodes are documented [here](docs/nodes_img.md). Use the above ToC to access them by category.

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
