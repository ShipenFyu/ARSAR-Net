import torch
from torch.fft import fft, ifft, fftshift, ifftshift


def ista_l1_one_object(image, echo, processor: dict, down_matrix, up_matrix, 
                       device_index, sparse_thr=200, length=5, res_thr=1e-10, maxiter=50):
    """
    L1 regularized ISTA implementation with GPU acceleration
    
    params:
        image (torch.Tensor): Initial image estimate
        echo (torch.Tensor): Input echo matrix
        processor (dict): System parameters dictionary
        down_matrix (torch.Tensor): Downsampling matrix
        up_matrix (torch.Tensor): Upsampling matrix
        device_index (str): Device index ('cuda' or 'cpu')
        sparse_thr (int): Sparsity threshold, the maximum number of non-zero points
        length (int): Iteration step
        res_thr (float): Convergence residual threshold
        maxiter (int): Maximum number of iterations
    
    return :
        torch.Tensor: Reconstrucation SAR image
    """
    device = torch.device(device_index if torch.cuda.is_available() else "cpu")
    echo = echo.to(device)
    
    # initialize
    nslow, nfast = image.shape
    rst = torch.zeros((nslow, nfast), dtype=torch.complex64, device=device)
    res = 1.0
    iter_num = 0
    
    while iter_num < maxiter and res > res_thr:
        rst_prev = rst.clone()
        
        residual = echo - echo_operator(rst, down_matrix, processor, device)
        rst = rst + imaging_operator(residual, up_matrix, processor, device) / length

        abs_rst = torch.abs(rst)
        flatten_abs = abs_rst.flatten()
        # sort in descending order, and take No.[sparse_thr] element as the threshold
        if len(flatten_abs) > sparse_thr:
            threshold = torch.sort(flatten_abs, descending=True).values[sparse_thr]
        else:
            threshold = 0.0

        rst = torch.where(abs_rst > threshold, 
                          (rst / (abs_rst + 1e-16)) * (abs_rst - threshold), 
                          torch.tensor(0.0, dtype=rst.dtype, device=device))

        numerator = torch.norm(rst - rst_prev, p='fro')
        denominator = torch.norm(rst_prev, p='fro')
        res = (numerator / (denominator + 1e-16)).item()

        iter_num += 1
    
    return rst


def admm_tv_one_object(image, echo, processor: dict, up_matrix, device_index, 
                       rho=0.5, eta=0.2, res_thr=1e-10, maxiter=50):
    """
    TV regularized ADMM implementation with GPU acceleration
    
    params:
        image (torch.Tensor): Initial image estimate
        echo (torch.Tensor): Input echo matrix (measurements)
        processor (dict): System parameters dictionary
        down_matrix (torch.Tensor): Downsampling matrix
        up_matrix (torch.Tensor): Upsampling matrix
        device_index (str): Device index ('cuda' or 'cpu')
        lambda_tv (float): TV regularization parameter
        rho (float): ADMM penalty parameter
        res_thr (float): Convergence threshold
        maxiter (int): Maximum number of iterations
    
    return:
        torch.Tensor: Reconstructed SAR image
    """
    device = torch.device(device_index if torch.cuda.is_available() else "cpu")
    echo = echo.to(device)
    
    # Initialize variables
    nslow, nfast = image.shape
    x = torch.zeros((nslow, nfast), dtype=torch.complex64, device=device)
    z = torch.zeros((nslow, nfast), dtype=torch.complex64, device=device)  # Auxiliary variable for gradient
    u = torch.zeros((nslow, nfast), dtype=torch.complex64, device=device)  # Dual variable 
    
    trivial_value = imaging_operator(echo, up_matrix, processor, device)
    iter_num = 0
    res = 1
    
    while iter_num < maxiter and res > res_thr:
        # 1. x-update (Data reconstruction step)
        x_prev = x.clone()
        penalty = rho * (z - u) -rho * x
        x = trivial_value + penalty
        
        # 2. z-update (TV proximal operator)
        tv_processor = TotalVarProces(iteration=6)
        z = tv_processor.forward(x, u)
        
        # 3. Dual variable update
        u += eta * (x - z)
        
        # Check convergence
        numerator = torch.norm(x - x_prev, p='fro')
        denominator = torch.norm(x_prev, p='fro')
        res = (numerator / (denominator + 1e-16)).item()
        iter_num += 1
            
    return x


def imaging_operator(echo, up_matrix, processor, device):
    '''
    Nonuniform Azimuth FFT -> RCMC -> Range FFT -> Range Compression
    -> Range IFFT -> Azimuth Compression -> Azimuth IFFT 
    '''
    rec_echo = torch.einsum('ij,jk->ik', up_matrix, echo)
    echo_fa = fftshift(fft(rec_echo, dim=0, norm='ortho'), dim=0)
    echo_fa = echo_fa * processor['sc'].to(device)
    echo_far = fftshift(fft(echo_fa, dim=1, norm='ortho'), dim=1)
    echo_far = echo_far * processor['rc'].to(device)
    echo_fa = ifft(ifftshift(echo_far, dim=1), dim=1, norm='ortho')
    echo_fa = echo_fa * processor['ac'].to(device)
    image = ifft(ifftshift(echo_fa, dim=0), dim=0, norm='ortho')
    
    return image


def echo_operator(image, down_matrix, processor, device):
    '''
    Azimuth FFT -> Conjugate Azimuth CompressionRCMC -> Range FFT 
    -> Conjugate Range Compression -> Range IFFT -> Conjugate RCMC -> Azimuth IFFT 
    '''
    image_fr = fftshift(fft(image, dim=0, norm='ortho'), dim=0)
    image_fr = image_fr * torch.conj(processor['ac'].to(device))
    image_fra = fftshift(fft(image_fr, dim=1, norm='ortho'), dim=1)
    image_fra = image_fra * torch.conj(processor['rc'].to(device))
    image_fr = ifft(ifftshift(image_fra, dim=1), dim=1, norm='ortho')
    image_fr = image_fr * torch.conj(processor['sc'].to(device))
    image_fr = ifft(ifftshift(image_fr, dim=0), dim=0, norm='ortho')
    echo = torch.einsum('ij,jk->ik', down_matrix, image_fr)
    
    return echo


class TotalVarProces:
    def __init__(self, iteration):
        self.iteration = iteration

        self.miu_1 = 0.5
        self.miu_2 = 0.5

    def tv_gradient(self, z):
        width_dif = torch.zeros_like(z)
        height_dif = torch.zeros_like(z)

        width_dif_s = torch.zeros_like(z)
        height_dif_s = torch.zeros_like(z)

        # wrap-around
        width_dif[:, :-1] = z[:, 1:] - z[:, :-1]
        width_dif[:, -1] = z[:, 0] - z[:, -1]

        height_dif[:-1, :] = z[1:, :] - z[:-1, :]
        height_dif[-1, :] = z[0, :] - z[-1, :]

        # second order differential
        width_dif_s[:, :-1] = width_dif[:, 1:] - width_dif[:, :-1]
        width_dif_s[:, -1] = width_dif[:, 0] - width_dif[:, -1]

        height_dif_s[:-1, :] = height_dif[1:, :] - height_dif[:-1, :]
        height_dif_s[-1, :] = height_dif[0, :] - height_dif[-1, :]

        grad = torch.zeros_like(z)
        grad = 2 * (width_dif_s + height_dif_s)

        return grad
    
    def additional_update(self, z, residual, grad):
        mid_value = torch.add(self.miu_1 * z, self.miu_2 * residual)

        return torch.sub(mid_value, grad)
    
    def forward(self, x, beta):
        residual = torch.add(x, beta)
        z = residual

        for _ in range(self.iteration):
            grad = self.tv_gradient(z)
            z = self.additional_update(z, residual, grad)

        return z
