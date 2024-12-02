import numpy as np
from skimage.metrics import structural_similarity


def normalize(img):
    """
    Normalize every pixel value in input image into 0-1
    """
    num_dim = len(img.shape)
    if num_dim >= 3:
        # product dim: batch_size, channel, etc. Remain Height and Width
        num_image = np.prod(img.shape[0:-2])
    elif num_dim == 2:
        num_image = 1
    else:
        raise ValueError(f'The shape of input image {img.shape} is wrong!')
    
    img = np.reshape(img, (num_image, img.shape[-2], img.shape[-1]))
    eps = 1e-15

    img_norm = np.empty_like(img)
    for i in range(num_image):
        img_norm[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min() + eps)
    img_norm = np.reshape(img_norm, img.shape)  # back to origin shape
    
    return img_norm


def psnr_evaluate(img, rec):
    sqrError = np.abs(img - rec) ** 2
    N = np.prod(img.shape[-2:])
    mse = np.sum(sqrError, axis=(-1, -2)) / N

    maxval = np.max(img, axis=(-1, -2)) + 1e-15
    psnr = 10 * np.log10(maxval ** 2 / (mse + 1e-15))
    
    return psnr


def ssim_evaluate(img, rec):
    num = img.shape[0]
    ssim = np.empty(num, dtype=np.float32)

    for i in range(num):
        ssim[i] = structural_similarity(img[i], rec[i], data_range=img[i].max())
    
    return ssim