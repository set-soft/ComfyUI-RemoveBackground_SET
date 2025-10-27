#
# Adapted from: https://github.com/kijai/ComfyUI-DepthAnythingV2
# Credits go to Kijai (Jukka Seppänen) https://github.com/kijai
#
# Adapted by Salvador E. Tropea
# Why?
# - Can automagically used when no depth maps are provided for the PDFNet model
# - Don't like the silent downloader (no progress and cryptic names)
# - Extra dependencies we can avoid, IDK why for "accelerate" if the code explicitly makes it optional
#
from comfy_api.latest import io
from . import CATEGORY_LOAD, CATEGORY_ADV
from .dan import KNOWN_MODELS, load_dan, run_dan

MODEL_TOOLTIP = ("The name of the model to use.\n"
                 "Small, Base and Large are available in 16 and 32 bits.\n"
                 "The 16 bits version works quite well for most uses.\n"
                 "PDFNet was trained using the Base version.")
DAN_MODEL_OUT_TOOLTIP = "The model ready to be used by the `Depth Anything V2` node"
DAModel = io.Custom("DAMODEL")
BATCHED_OPS = io.Int.Input("batch_size", default=1, min=1, max=256, step=1, tooltip="How many images to process at once")
DEPTH_IMG_TOOLTIP = ("The same map in a format compatible with nodes that needs an image.\n"
                     "The three channels (R, G, B) are the same buffer, shared with the mask.")


class DownloadAndLoadDepthAnythingV2Model(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DownloadAndLoadDepthAnythingV2Model_SET",
            display_name="Load Depth Anything by name",
            category=CATEGORY_LOAD,
            description=("Load a Depth Anything V2 model.\n"
                         "Models are autodownload to `ComfyUI/models/depthanything` from\n"
                         "https://huggingface.co/Kijai/DepthAnythingV2-safetensors/tree/main\n\n"
                         "F16 might reduce quality, be careful."),
            inputs=[io.Combo.Input("model", options=list(KNOWN_MODELS.keys()), tooltip=MODEL_TOOLTIP)],
            outputs=[DAModel.Output(display_name="da_v2_model", tooltip=DAN_MODEL_OUT_TOOLTIP)]
        )

    @classmethod
    def execute(cls, model) -> io.NodeOutput:
        return io.NodeOutput(load_dan(model))


class DepthAnything_V2(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DepthAnything_V2_SET",
            display_name="Depth Anything V2",
            category=CATEGORY_ADV,
            description="Create a depth map of the image\nSee: https://depth-anything-v2.github.io",
            inputs=[
                DAModel.Input("da_model", tooltip="The model from the `Load Depth Anything by name` node."),
                io.Image.Input("images", tooltip="One or more images to process, will be normalized and scaled."),
                BATCHED_OPS],
            outputs=[io.Mask.Output(display_name="depths", tooltip="The depth map"),
                     io.Image.Output(display_name="depth_imgs", tooltip=DEPTH_IMG_TOOLTIP)]
        )

    @classmethod
    def execute(cls, da_model, images, batch_size) -> io.NodeOutput:
        return io.NodeOutput(*run_dan(da_model, images, batch_size))
