#
# ESNet: Evolution and Succession Network for High-Resolution Salient Object Detection
#
# Hongyu Liu, Runmin Cong, Hua Li, Qianqian Xu, Qingming Huang, Wei Zhang
# https://openreview.net/pdf?id=SERrqPDvoY
#
# https://github.com/big-feather/ESNet_ICML24
#
# Utility kernels
#
import torch.nn as nn


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    # x_coord = torch.arange(kernel_size)
    # x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    # y_grid = x_grid.t()
    # xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    # mean = (kernel_size - 1) / 2.
    # variance = sigma ** 2.

    # # Calculate the 2-dimensional gaussian kernel which is
    # # the product of two gaussian distributions for two different
    # # variables (in this case called x and y)
    # gaussian_kernel = ((1. / (2. * math.pi * variance)) *
    #                    torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)))

    # # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # # Reshape to 2d depthwise convolutional weight
    # gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    # gaussian_filter.weight.data = gaussian_kernel
    # gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def get_sobel_kernel(channels=1):
    kernel_size = 3
    sobel_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                             bias=False, padding=kernel_size // 2)

    # snp = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # sobel_kernel = torch.from_numpy(snp)
    # sobel_kernel = sobel_kernel.view(1, 1, kernel_size, kernel_size)
    # sobel_kernel = sobel_kernel.repeat(channels, 1, 1, 1)

    # sobel_filter.weight.data = sobel_kernel
    # sobel_filter.weight.requires_grad = False

    return sobel_filter
