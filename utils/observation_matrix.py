import torch
import numpy as np

from utils.config import Nslow, Nfast, device_index

device = torch.device(device_index if torch.cuda.is_available() else "cpu")


def get_fourier_matrix(dim, down_dim):
    '''
    generate Fourier transform matrix with fft_shift
    '''
    n = torch.arange(dim)
    k = torch.arange(0, 1, 1 / down_dim).view(-1, 1)
    jTwoPi = torch.tensor(1j * 2 * np.pi, dtype=torch.complex64)  # 1j * 2π
    scale = torch.tensor(1. / np.sqrt(dim), dtype=torch.complex64)  # 1/sqrt(N)

    Ah = torch.exp(-jTwoPi * (k - 1 / 2) * n) * scale
    Aht = Ah.conj().permute(1, 0)

    return Ah, Aht


def get_ob_matrix(batch_size):
    Ah_fast, Aht_fast = get_fourier_matrix(Nfast, Nfast)
    Ah_slow, Aht_slow = get_fourier_matrix(Nslow, Nslow)

    Phi_fast = torch.matmul(Ah_fast, Aht_fast).expand(batch_size, -1, -1).to(device)
    Phi_slow = torch.matmul(Aht_slow, Ah_slow).expand(batch_size, -1, -1).to(device)

    return Phi_fast, Phi_slow


def get_kronecker_matrix(batch_size):
    F_fast, Ft_fast = get_fourier_matrix(Nfast, Nfast)
    F_slow, Ft_slow = get_fourier_matrix(Nslow, Nslow)

    Phi_fast = torch.matmul(F_fast, Ft_fast).expand(batch_size, -1, -1)
    Phi_slow = torch.matmul(Ft_slow, F_slow).expand(batch_size, -1, -1)

    Phi_fast_H = Phi_fast.conj().permute(0, 2, 1)
    Phi_slow_H = Phi_slow.conj().permute(0, 2, 1)

    Phi_fast_a = torch.matmul(Phi_fast, Phi_fast_H)  # \Phi_R * \Phi_R^H
    Phi_slow_a = torch.matmul(Phi_slow_H, Phi_slow)  # \Phi_L^H * \Phi_L

    Phi_fast_at = Phi_fast_a.permute(0, 2, 1)
    kronecker_matrix = torch.kron(Phi_fast_at, Phi_slow_a).to(device)
    
    return kronecker_matrix
