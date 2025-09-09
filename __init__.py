from .src.nodes import main_logger, __version__
import folder_paths
import os
from seconohe.register_nodes import register_nodes
from seconohe import JS_PATH


models_dir_key = "birefnet"
models_dir_default = os.path.join(folder_paths.models_dir, "BiRefNet")
if models_dir_key not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths[models_dir_key] = (
        [os.path.join(folder_paths.models_dir, "BiRefNet")], folder_paths.supported_pt_extensions)
else:
    if not os.path.exists(models_dir_default):
        os.makedirs(models_dir_default, exist_ok=True)
    folder_paths.add_model_folder_path(models_dir_key, models_dir_default)


# Done here because we need to first register "birefnet"
from .src.nodes import birefnetNode  # noqa: E402


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes(main_logger, [birefnetNode], version=__version__)
WEB_DIRECTORY = JS_PATH
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
