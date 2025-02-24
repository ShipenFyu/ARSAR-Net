# -----------------------------------------------------------------------------------------------------------
# Cited from: Nonsparse SAR Scene Imaging Network Based on Sparse Representation and Approximate Observations
# -----------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.fft import fft, ifft, fftshift, ifftshift


def soft_threshold_complex(x, threshold):
    soft_x = (x/torch.abs(x)) * torch.maximum(torch.abs(x) - threshold, torch.tensor(0.0, device=x.device))
    return soft_x


class SRNet(nn.Module):
    def __init__(self, processor, device, up_matrix, num_layer, mode):
        super().__init__()
        self.device = device
        self.num_layer = num_layer

        self.layer_list = nn.ModuleList([BasicBlock(processor, device, up_matrix, mode) 
                                       for _ in range(num_layer)])

    def forward(self, echo_downsampling):
        '''
        echo_downsampling ----> S_d
        matrix_downsampling ----> ψ_a
        image ----> origin image zeros matrix
        '''
        batch_size, _, width = echo_downsampling.shape
        xi = torch.zeros(batch_size, width, width).to(self.device)

        for num in range(self.num_layer):
            xi = self.layer_list[num](echo_downsampling, xi)

        return xi
    

class BasicBlock(nn.Module):
    def __init__(self, processor, device, up_matrix, mode):
        super().__init__()
        self.processor = processor
        self.device = device
        self.up_matrix = up_matrix
        self.mode = mode
        self.net_object()

    def net_object(self):
        self.r_net = RLinearityModule(self.processor, self.device, self.up_matrix)

        if self.mode == 'plus':
            self.n_net = NPlusNonlinearityModule(device=self.device)
        else:
            assert self.mode == 'basic'
            self.n_net = NNonlinearityModule(device=self.device)

    def forward(self, echo_downsampling, xi):
        r = self.r_net(echo_downsampling, xi)
        xi_next = self.n_net(r)
        
        return xi_next


class RLinearityModule(nn.Module):
    def __init__(self, processor, device, up_matrix):
        super().__init__()
        self.processor = processor
        self.device = device
        self.up_matrix = up_matrix
        self.down_matrix = torch.transpose(up_matrix, dim0=0, dim1=1)
        self.miu = nn.Parameter(data=torch.tensor(1.0), requires_grad=True).to(device)

    def inverse_imaging_operator(self, image):
        '''
        Azimuth FFT -> Conjugate Azimuth Compression(ac) -> Range FFT 
        -> Conjugate Range compression(rc) -> Range IFFT -> Conjugate RCMC(sc) -> Azimuth IFFT
        '''
        sc_conjugate = torch.conj(self.processor["sc"])
        rc_conjugate = torch.conj(self.processor["rc"])
        ac_conjugate = torch.conj(self.processor["ac"])

        image = fftshift(fft(image, dim=-2, norm='ortho'), dim=-2)
        image_ac = image * ac_conjugate.to(self.device)
        image_ac = fftshift(fft(image_ac, dim=-1, norm='ortho'), dim=-1)
        image_ac_rc = image_ac * rc_conjugate.to(self.device)
        image_ac_rc = ifft(ifftshift(image_ac_rc, dim=-1), dim=-1, norm='ortho')
        image_ac_rc_sc = image_ac_rc * sc_conjugate.to(self.device)
        echo = ifft(ifftshift(image_ac_rc_sc, dim=-2), dim=-2, norm='ortho')

        return echo
    
    def imaging_operator(self, echo):
        '''
        Azimuth FFT -> RCMC(sc) -> Range FFT -> Range Compression(rc)
        -> Range IFFT -> Azimuth Compression(ac) -> Azimuth IFFT 
        '''
        sc = self.processor["sc"]
        rc = self.processor["rc"]
        ac = self.processor["ac"]

        echo = fftshift(fft(echo, dim=-2, norm='ortho'), dim=-2)
        echo_sc = echo * sc.to(self.device)
        echo_sc = fftshift(fft(echo_sc, dim=-1, norm='ortho'), dim=-1)
        echo_sc_rc = echo_sc * rc.to(self.device)
        echo_sc_rc = ifft(ifftshift(echo_sc_rc, dim=-1), dim=-1, norm='ortho')
        echo_sc_rc_ac = echo_sc_rc * ac.to(self.device)
        image = ifft(ifftshift(echo_sc_rc_ac, dim=-2), dim=-2, norm='ortho')

        return image
        
    def forward(self, echo_downsampling, xi):
        echo_downsampling = echo_downsampling
        echo_res = echo_downsampling - self.down_matrix @ self.inverse_imaging_operator(xi)
        echo = self.up_matrix @ echo_res
        r = xi + self.miu * self.imaging_operator(echo)

        return r
    

class NNonlinearityModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.t = nn.Parameter(data=torch.tensor(1.0), requires_grad=True).to(device)
        # the same transformation share the same weight
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.conv1_hat = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1)
        self.conv2_hat = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), stride=1, padding=1)
        self.b = nn.BatchNorm2d(num_features=32)
        self.b_hat = nn.BatchNorm2d(num_features=32)
        self.activation = nn.ReLU()
        self.activation_hat = nn.ReLU()

    def forward(self, r):
        r = torch.unsqueeze(r, 1)

        r_real = torch.real(r)
        r_image = torch.imag(r)

        r_real_c1 = self.conv1(r_real)
        r_real_c2 = self.conv2(self.activation(self.b(r_real_c1)))
        r_image_c1 = self.conv1(r_image)
        r_image_c2 = self.conv2(self.activation(self.b(r_image_c1)))

        r_transformed = torch.complex(real=r_real_c2, imag=r_image_c2)
        r_soft = soft_threshold_complex(x=r_transformed, threshold=self.t)
        
        r_hat_real_c1 = self.conv1_hat(torch.real(r_soft))
        r_hat_real_c2 = self.conv2_hat(self.activation_hat(self.b_hat(r_hat_real_c1)))
        r_hat_image_c1 = self.conv1_hat(torch.imag(r_soft))
        r_hat_image_c2 = self.conv2_hat(self.activation_hat(self.b_hat(r_hat_image_c1)))

        xi = torch.complex(r_hat_real_c2, r_hat_image_c2)
        xi = torch.squeeze(xi, 1)

        return xi


class NPlusNonlinearityModule(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.t = nn.Parameter(data=torch.tensor(1.0), requires_grad=True).to(device)
        # the same transformation(Real part & Image part) share the same weight
        self.conv_d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1)
        self.conv1_hat = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        self.conv2_hat = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.conv_g = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3,3), stride=1, padding=1)
        self.b = nn.BatchNorm2d(num_features=64)
        self.b_hat = nn.BatchNorm2d(num_features=64)
        self.activation = nn.ReLU()
        self.activation_hat = nn.ReLU()

    def forward(self, r):
        r = torch.unsqueeze(r, 1)

        r_real = torch.real(r)
        r_image = torch.imag(r)
        r_real_c1 = self.conv1(self.conv_d(r_real))
        r_real_c2 = self.conv2(self.activation(self.b(r_real_c1)))
        r_image_c1 = self.conv1(self.conv_d(r_image))
        r_image_c2 = self.conv2(self.activation(self.b(r_image_c1)))

        r_transformed = torch.complex(real=r_real_c2, imag=r_image_c2)
        r_soft = soft_threshold_complex(r_transformed, threshold=self.t)
        
        r_hat_real_c1 = self.conv1_hat(torch.real(r_soft))
        r_hat_real_c2 = self.conv2_hat(self.activation_hat(self.b_hat(r_hat_real_c1)))
        r_hat_image_c1 = self.conv1_hat(torch.imag(r_soft))
        r_hat_image_c2 = self.conv2_hat(self.activation_hat(self.b_hat(r_hat_image_c1)))
        
        xi_real = self.conv_g(r_hat_real_c2) + r_real
        xi_image = self.conv_g(r_hat_image_c2) + r_image

        xi = torch.complex(xi_real, xi_image)
        xi = torch.squeeze(xi, 1)

        return xi
