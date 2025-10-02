from .wrapper import MobileNetV2Backbone


# ------------------------------------------------------------------------------
#  Replaceable Backbones
# ------------------------------------------------------------------------------

SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
}
