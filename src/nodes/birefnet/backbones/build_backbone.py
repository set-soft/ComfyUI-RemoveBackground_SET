from .swin_v1 import swin_v1_t, swin_v1_l


def build_backbone(bb_name, pretrained=True, params_settings=''):
    if bb_name == 'swin_v1_t':
        bb = swin_v1_t()
    elif bb_name == 'swin_v1_l':
        bb = swin_v1_l()
    else:
        raise ValueError('Unsupported back bone')
    return bb
