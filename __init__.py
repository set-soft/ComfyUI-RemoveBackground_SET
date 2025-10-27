from .src.nodes import MODELS_DIR_KEY, MODELS_DIR, main_logger, __version__
from seconohe.register_nodes import register_nodes_v3, register_models_key, check_v3
from seconohe import JS_PATH
WEB_DIRECTORY = JS_PATH

register_models_key(main_logger, MODELS_DIR_KEY, MODELS_DIR, {"birefnet": "BiRefNet", "rembg": "rembg"})
check_v3(main_logger)


async def comfy_entrypoint():
    from .src.nodes import nodes
    from .src.nodes import nodes_dan
    return register_nodes_v3(main_logger, [nodes, nodes_dan], version=__version__)
