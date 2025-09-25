import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.morphology import dilation, erosion
from torch.nn.parameter import Parameter


class ModuleWithConstantKernel(nn.Module):
    """ An nn.Module that contains a self.kernel buffer that is a constant and won't be loaded from the state_dict.
        This is needed because the checkpoints doesn't contain these values. """
    def __init__(self, kernel):
        super().__init__()
        self.register_buffer('kernel', kernel)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """ Make `kernel` buffer a constant """
        kernel_key = prefix + 'kernel'
        if kernel_key in state_dict:
            # If the state_dict contains it just discard the value, currently isn't the case
            del state_dict[kernel_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if kernel_key in missing_keys:
            # If the key is missing avoid an error, currently this IS the case
            missing_keys.remove(kernel_key)


class ImagePyramid(ModuleWithConstantKernel):
    def __init__(self):
        self.ksize = 7
        self.sigma = 1
        self.channels = 1

        # Original code:
        # k = cv2.getGaussianKernel(ksize, sigma)
        # k = np.outer(k, k)
        # k = torch.tensor(k).float()
        # self.kernel = k.repeat(channels, 1, 1, 1)

        kernel = torch.tensor([[[[1.9652e-05, 2.3941e-04, 1.0730e-03, 1.7690e-03, 1.0730e-03, 2.3941e-04, 1.9652e-05],
                                 [2.3941e-04, 2.9166e-03, 1.3071e-02, 2.1551e-02, 1.3071e-02, 2.9166e-03, 2.3941e-04],
                                 [1.0730e-03, 1.3071e-02, 5.8582e-02, 9.6585e-02, 5.8582e-02, 1.3071e-02, 1.0730e-03],
                                 [1.7690e-03, 2.1551e-02, 9.6585e-02, 1.5924e-01, 9.6585e-02, 2.1551e-02, 1.7690e-03],
                                 [1.0730e-03, 1.3071e-02, 5.8582e-02, 9.6585e-02, 5.8582e-02, 1.3071e-02, 1.0730e-03],
                                 [2.3941e-04, 2.9166e-03, 1.3071e-02, 2.1551e-02, 1.3071e-02, 2.9166e-03, 2.3941e-04],
                                 [1.9652e-05, 2.3941e-04, 1.0730e-03, 1.7690e-03, 1.0730e-03, 2.3941e-04, 1.9652e-05]]]],
                              dtype=torch.float32)
        super().__init__(kernel)

    def expand(self, x):
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel * 4, groups=self.channels)
        return x

    def reduce(self, x):
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel, groups=self.channels)
        x = x[:, :, ::2, ::2]
        return x

    def deconstruct(self, x):
        reduced_x = self.reduce(x)
        expanded_reduced_x = self.expand(reduced_x)

        if x.shape != expanded_reduced_x.shape:
            expanded_reduced_x = F.interpolate(expanded_reduced_x, x.shape[-2:])

        laplacian_x = x - expanded_reduced_x
        return reduced_x, laplacian_x

    def reconstruct(self, x, laplacian_x):
        expanded_x = self.expand(x)
        if laplacian_x.shape != expanded_x:
            laplacian_x = F.interpolate(laplacian_x, expanded_x.shape[-2:], mode='bilinear', align_corners=True)
        return expanded_x + laplacian_x


class Transition(ModuleWithConstantKernel):
    def __init__(self, k=3):
        # Original code:
        # self.kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).float()

        # Use the pre-computed, 100% accurate kernels from OpenCV
        if k == 5:
            kernel = torch.tensor([[0., 0., 1., 0., 0.],
                                   [1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1.],
                                   [0., 0., 1., 0., 0.]], dtype=torch.float32)
        elif k == 9:
            kernel = torch.tensor([[0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [0., 0., 0., 0., 1., 0., 0., 0., 0.]], dtype=torch.float32)
        elif k == 17:
            kernel = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                                   [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
                                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.],
                                   [0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0.],
                                   [0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.],
                                   [0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.],
                                   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported kernel size: {k}. Only 5, 9, and 17 are supported.")

        super().__init__(kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_prob = torch.sigmoid(x)

        dx = dilation(x_prob, self.kernel)
        ex = erosion(x_prob, self.kernel)

        gradient = dx - ex
        return torch.where(gradient > 0.5, 1.0, 0.0)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same', bias=False,
                 bn=True, relu=False):
        super(Conv2d, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.reset_parameters()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(SelfAttention, self).__init__()

        self.mode = mode

        self.query_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.key_conv = Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1))
        self.value_conv = Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.softmax(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out
