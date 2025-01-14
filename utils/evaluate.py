import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.ndimage import gaussian_filter


def normalize(images):
    """
    Perform histogram equalization on images with shape: [batch_size, height, width]
    """
    images = np.asarray(images)
    batch_size, height, width = images.shape
    equalized_images = np.zeros_like(images, dtype=np.uint8)

    for i in range(batch_size):
        img = images[i]
        
        img_normalized = (img - img.min()) / (img.max() - img.min()) * 255
        img_normalized = img_normalized.astype(np.uint8)
        
        hist, bins = np.histogram(img_normalized.flatten(), bins=256, range=[0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]  # normalize to [0, 1]
        
        equalized_img = np.interp(img_normalized.flatten(), bins[:-1], cdf_normalized * 255)
        equalized_images[i] = equalized_img.reshape(height, width)

    return equalized_images


def range_cut(image):
    return np.clip(image, 0, 255)


def psnr_evaluate(image, rec):
    image = np.asarray(image)
    rec = np.asarray(rec)
    sqrError = np.abs(image - rec) ** 2
    N = np.prod(image.shape[-2:])
    mse = np.sum(sqrError, axis=(-1, -2)) / N

    maxval = np.max(image, axis=(-1, -2)) + 1e-15
    psnr = 10 * np.log10(maxval ** 2 / (mse + 1e-15))
    
    return psnr


def ssim_evaluate(image, rec, k1=0.01, k2=0.03, L=255, sigma=1.5):
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


def aliasing_construct(echo, down_matrix, operator):
    rec_echo = np.einsum('ij,bjk->bik', down_matrix, echo)
    echo_fa = fftshift(fft(rec_echo, axis=1, norm='ortho'), axes=1)
    echo_fa = echo_fa * operator['sc'].numpy()
    echo_far = fftshift(fft(echo_fa, axis=2, norm='ortho'), axes=2)
    echo_far = echo_far * operator['rc'].numpy()
    echo_fa = ifft(ifftshift(echo_far, axes=2), axis=2, norm='ortho')
    echo_fa = echo_fa * operator['ac'].numpy()
    image = ifft(ifftshift(echo_fa, axes=1), axis=1, norm='ortho')
    
    return image
