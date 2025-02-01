import torch
import torch.nn.functional as F
from math import pi


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Converts a RGB image to XYZ.

    See :class:`~kornia.color.RgbToXyz` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to XYZ.

    Returns:
        torch.Tensor: XYZ version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    x: torch.Tensor = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y: torch.Tensor = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z: torch.Tensor = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out: torch.Tensor = torch.stack((x, y, z), -3)

    return out

def xyz_to_lab(image: torch.Tensor) -> torch.Tensor:

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    x: torch.Tensor = image[..., 0, :, :]
    y: torch.Tensor = image[..., 1, :, :]
    z: torch.Tensor = image[..., 2, :, :]

    l: torch.Tensor = (116 * x) - 16
    a: torch.Tensor = 500 * (x - y)
    b: torch.Tensor = 200 * (y - z)

    out: torch.Tensor = torch.stack((l, a, b), -3)

    return out

def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac / v

    s[torch.isnan(s)] = 0.

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0

    h = 2 * pi * h
    return torch.stack([h, s, v], dim=-3)   


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to LAB.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))
    

    return xyz_to_lab(rgb_to_xyz(image))

def rgb_to_grayscale(input: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to grayscale.

    See :class:`~kornia.color.RgbToGrayscale` for details.

    Args:
        input (torch.Tensor): RGB image to be converted to grayscale.

    Returns:
        torch.Tensor: Grayscale version of the image.
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if len(input.shape) < 3 and input.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(input.shape))

    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    gray: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return gray

def gaussian_kernel(window_size, sigma):
    """
    Create a 1D Gaussian kernel.
    """
    gauss = torch.arange(window_size).float() - (window_size - 1) / 2
    gauss = torch.exp(-(gauss ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Create a 2D Gaussian window for SSIM calculation.
    """
    _1D_window = gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.T
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    """
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)

    # Mean of each image
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # Variance and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2 ** 2
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1 * mu2

    # SSIM calculation
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return ssim_map.mean()

def calculate_ssim_torch(image1, image2):
    """
    Wrapper function to calculate SSIM between two images using PyTorch.
    """
    # Ensure images are in the same format
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    # Add batch dimension if necessary
    if len(image1.shape) == 3:
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)

    return ssim(image1, image2)