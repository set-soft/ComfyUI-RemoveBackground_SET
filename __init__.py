from .src.nodes import MODELS_DIR_KEY, MODELS_DIR, main_logger, __version__
import folder_paths
import os
from seconohe.register_nodes import register_nodes_v3
from seconohe import JS_PATH


models_dir_default = os.path.join(folder_paths.models_dir, MODELS_DIR)
if MODELS_DIR_KEY not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[MODELS_DIR_KEY] = (
        [os.path.join(folder_paths.models_dir, MODELS_DIR)], folder_paths.supported_pt_extensions)
else:
    if not os.path.exists(models_dir_default):
        os.makedirs(models_dir_default, exist_ok=True)
    folder_paths.add_model_folder_path(MODELS_DIR_KEY, models_dir_default)


# Done here because we need to first register "birefnet"
from .src.nodes import nodes  # noqa: E402
from .src.nodes import nodes_dan  # noqa: E402
WEB_DIRECTORY = JS_PATH


async def comfy_entrypoint():
    return register_nodes_v3(main_logger, [nodes, nodes_dan], version=__version__)
