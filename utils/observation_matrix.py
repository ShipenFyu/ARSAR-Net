import torch
import numpy as np

from utils.config import p, Nslow, Nfast, device_index

device = torch.device(device_index if torch.cuda.is_available() else "cpu")


def get_fourier_matrix(dim):
    '''
    generate Fourier transform matrix with fft_shift
    '''
    n = torch.arange(dim)
    k = torch.arange(0, 1, 1 / dim).view(-1, 1)
    jTwoPi = torch.tensor(1j * 2 * np.pi, dtype=torch.complex64)  # 1j * 2π
    scale = torch.tensor(1. / np.sqrt(dim), dtype=torch.complex64)  # 1/sqrt(N)

    Ah = torch.exp(-jTwoPi * (k - 1 / 2) * n) * scale
    Aht = Ah.conj().permute(1, 0)

    return Ah, Aht


def get_ob_matrix(batch_size):
    Ah_fast, Aht_fast = get_fourier_matrix(Nfast)
    Ah_slow, Aht_slow = get_fourier_matrix(Nslow)

    Phi_fast = torch.matmul(Ah_fast, Aht_fast).expand(batch_size, -1, -1).to(device)
    Phi_slow = torch.matmul(Aht_slow, Ah_slow).expand(batch_size, -1, -1).to(device)

    operator = torch.conj(p['ac']) * torch.conj(p['rc']) * torch.conj(p['sc'])
    operator = operator.to(torch.complex64)

    return Phi_fast, Phi_slow, operator.to(device)
