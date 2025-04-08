import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.fft import fft, ifft, fftshift, ifftshift


def range_cut(image):
    return torch.clamp(image, 0, 255)


def nrmse_evaluate(image, rec):
    image = torch.as_tensor(image)
    rec = torch.as_tensor(rec)
    batch_size = image.shape[0]

    residual = rec - image
    loss = 0
    for index in range(batch_size):
        residual_norm = torch.linalg.matrix_norm(residual[index], ord=2)
        target_norm = torch.linalg.matrix_norm(image[index], ord=2)
        loss += residual_norm / target_norm 
    loss_average = loss / batch_size

    return loss_average


def psnr_evaluate(image, rec):
    image = torch.as_tensor(image)
    rec = torch.as_tensor(rec)
    sqr_error = (image - rec).pow(2)
    n = image.shape[-1] * image.shape[-2]
    mse = sqr_error.sum(dim=(-1, -2)) / n

    maxval = image.amax(dim=(-1, -2)) + 1e-15
    psnr = 10 * torch.log10(maxval ** 2 / (mse + 1e-15))

    return psnr


def psnr_mean(psnr):
    '''
    Optional function for psnr mean
    '''
    psnr = torch.as_tensor(psnr)
    psnr_p = 10 ** (psnr / 10)

    return 10 * torch.log10(psnr_p.mean())


def ssim_evaluate(image, rec, k1=0.01, k2=0.03, L=255, sigma=1.5):
    '''
    SSIM evaluation in Numpy with gaussian filter
    '''
    image = np.asarray(image)
    rec = np.asarray(rec)
    batch_size = image.shape[0]
    
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    ssim_values = np.zeros(batch_size)
    for i in range(batch_size):
        img1 = image[i]
        img2 = rec[i]
        
        mu1 = gaussian_filter(img1, sigma=sigma)
        mu2 = gaussian_filter(img2, sigma=sigma)
        sigma1 = gaussian_filter(img1 ** 2, sigma=sigma) - mu1 ** 2
        sigma2 = gaussian_filter(img2 ** 2, sigma=sigma) - mu2 ** 2
        sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1 * mu2

        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

        ssim_map = numerator / denominator
        ssim_values[i] = ssim_map.mean()

    return ssim_values


def gaussian_kernel(image, sigma):
    '''
    Gaussian filter by Pytorch
    '''
    image = image.unsqueeze(0)

    radius = int(4*sigma + 0.5)
    size = 2 * radius + 1
    x = torch.arange(-radius, radius+1, dtype=torch.float32, device=image.device)
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    padding = radius
        
    filtered = F.conv2d(
        input=image,
        weight=kernel_2d.view(1, 1, size, size).expand(image.size(1), 1, size, size),
        padding=padding,
    )
    
    return filtered.squeeze(0)


def ssim_evaluate_tensor(image, rec, k1=0.01, k2=0.03, L=255, sigma=1.5):
    device = image.device
    image = torch.as_tensor(image, device=device)
    rec = torch.as_tensor(rec, device=device)
    
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    batch_size = image.shape[0]
    ssim_values = torch.zeros(batch_size, device=device)
    
    for index in range(batch_size):
        img1 = image[index]
        img2 = rec[index]

        mu1 = gaussian_kernel(img1, sigma)
        mu2 = gaussian_kernel(img2, sigma)
        
        sigma1_sq = gaussian_kernel(img1**2, sigma) - mu1**2
        sigma2_sq = gaussian_kernel(img2**2, sigma) - mu2**2
        sigma12 = gaussian_kernel(img1 * img2, sigma) - mu1 * mu2
        
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_p = numerator / denominator
        ssim_values[index] = ssim_p.mean()
        
    return ssim_values


def aliasing_construct(echo, up_matrix, operator):
    echo = torch.as_tensor(echo)
    device = echo.device

    rec_echo = torch.einsum('ij,bjk->bik', up_matrix, echo)
    echo_fa = fftshift(fft(rec_echo, dim=1, norm='ortho'), dim=1)
    echo_fa = echo_fa * operator['sc'].to(device)
    echo_far = fftshift(fft(echo_fa, dim=2, norm='ortho'), dim=2)
    echo_far = echo_far * operator['rc'].to(device)
    echo_fa = ifft(ifftshift(echo_far, dim=2), dim=2, norm='ortho')
    echo_fa = echo_fa * operator['ac'].to(device)
    image = ifft(ifftshift(echo_fa, dim=1), dim=1, norm='ortho')
    
    return image
