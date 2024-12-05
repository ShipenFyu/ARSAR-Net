import torch
import torchpwl
import torch.nn as nn
from torch.fft import fft, ifft, fftshift, ifftshift

from utils.config import device_index

device = torch.device(device_index if torch.cuda.is_available() else "cpu")


class ADMMIRNet(nn.Module):
    def __init__(
        self, 
        operator,
        num_phase,
        iteration,
        in_channels = 1,
        out_channels = 32,
        kernel_size = 3, 
        num_breakpoints = 60,
        ):
        super(ADMMIRNet, self).__init__()
        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.eta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        self.reconstruction_start = ReconstructionLayer(self.rho, operator, is_first=True)
        self.reconstruction_end = ReconstructionLayer(self.rho, operator, is_first=False)
        self.multiple = MultipleLayer(self.eta, is_first=True)
        layers = []

        for _ in range(num_phase):
            layers.append(BasicBlock(operator, in_channels, out_channels, 
                                     kernel_size, num_breakpoints, iteration, self.rho, self.eta))
        
        self.iteration_net = nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.reconstruction_start(input, 0, 0)
        beta = self.multiple(0, x, 0)
        z = torch.zeros_like(x, device=device_index)

        input_dict = dict()
        input_dict['input'] = input
        input_dict['x'] = x
        input_dict['z'] = z
        input_dict['beta'] = beta

        input_dict = self.iteration_net(input_dict)
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction_end(input, z, beta)

        return x


class BasicBlock(nn.Module):
    def __init__(self, operator, in_channels, out_channels, 
                 kernel_size, num_breakpoints, iteration, rho, eta):
        super(BasicBlock, self).__init__()
        self.reconstruction = ReconstructionLayer(rho, operator)
        self.multiple = MultipleLayer(eta)
        self.recurrent = RecurrentBlock(in_channels, out_channels, kernel_size, 
                                        num_breakpoints, iteration)

    
    def forward(self, input_dict):
        input = input_dict['input']
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction(input, z, beta)
        z = self.recurrent(x, beta)
        beta = self.multiple(beta, x, z)

        input_dict['x'] = x
        input_dict['z'] = z
        input_dict['beta'] = beta

        return input_dict


class ReconstructionLayer(nn.Module):
    def __init__(self, rho, operator, is_first=False):
        super(ReconstructionLayer, self).__init__()
        self.rho = rho
        self.operator = operator
        self.is_first = is_first     

    def G_strip(self, img):
        img_fr = fftshift(fft(img, dim=1), dim=1)
        img_fr = img_fr * torch.conj(self.operator['ac']).to(device)
        img_fra = fftshift(fft(img_fr, dim=2), dim=2)
        img_fra = img_fra * torch.conj(self.operator['rc']).to(device)
        img_fr = ifft(ifftshift(img_fra, dim=2), dim=2)
        img_fr = img_fr * torch.conj(self.operator['sc']).to(device)
        echo = ifft(ifftshift(img_fr, dim=1), dim=1)
        return echo
    
    def I_strip(self, echo):
        echo_fr = fftshift(fft(echo, dim=1), dim=1)
        echo_fr = echo_fr * self.operator['sc'].to(device)
        echo_fra = fftshift(fft(echo_fr, dim=2), dim=2)
        echo_fra = echo_fra * self.operator['rc'].to(device)
        echo_fr = ifft(ifftshift(echo_fra, dim=2), dim=2)
        echo_fr = echo_fr * self.operator['ac'].to(device)
        image = ifft(ifftshift(echo_fr, dim=1), dim=1)
        return image

    def forward(self, input, z, beta):
        if self.is_first:
            trivial_value = self.I_strip(input)  # initialize

            return trivial_value
        else:
            iter_value = self.G_strip(self.rho * torch.sub(z, beta))
            iter_echo = torch.add(input, iter_value)
            trivial_value = self.I_strip(iter_echo)

            return trivial_value


class MultipleLayer(nn.Module):
    def __init__(self, eta, is_first=False):
        super(MultipleLayer, self).__init__()
        self.eta = eta
        self.is_first = is_first

    def forward(self, beta, x, z):
        if self.is_first:
            return torch.zeros_like(x, device=device_index)  # initialize
        else:
            return torch.add(beta, self.eta * torch.sub(x, z))


class RecurrentBlock(nn.Module):
    '''
    Cited from: "ADMM-CSNet: A Deep Learning Approach for Image Compressive Sensing"
    '''
    def __init__(self, in_channels, out_channels, kernel_size, num_breakpoints, iteration):
        super(RecurrentBlock, self).__init__()
        self.iteration = iteration

        self.miu_1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.miu_2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.conv_1 = ConvolutionNormalLayer(in_channels, out_channels, kernel_size)
        self.nonlinear = NonLinearLayer(out_channels, num_breakpoints)
        self.conv_2 = ConvolutionConjLayer(out_channels, in_channels, kernel_size)
        self.additional_start = AdditionalLayer(self.miu_1, self.miu_2, is_first=True)
        self.additional = AdditionalLayer(self.miu_1, self.miu_2)
        self.reset_parameters()  # initialize convolution layers' weights

    def reset_parameters(self):
        self.conv_1.conv.weight = torch.nn.init.normal_(self.conv_1.conv.weight, mean=0, std=1)
        self.conv_2.conv.weight = torch.nn.init.normal_(self.conv_2.conv.weight, mean=0, std=1)
        self.conv_1.conv.weight.data = self.conv_1.conv.weight.data * 0.025
        self.conv_2.conv.weight.data = self.conv_2.conv.weight.data * 0.025

    def forward(self, x, beta):
        z = self.additional_start(0, 0, x, beta)  # initialize

        for _ in range(self.iteration):
            z_channeled = torch.unsqueeze(z, 1)
            conv1 = self.conv_1(z_channeled)
            h = self.nonlinear(conv1)
            conv2 = self.conv_2(h)
            conv2_shaped = torch.squeeze(conv2, 1)
            z = self.additional(z, conv2_shaped, x, beta)
        
        return z
    

class ConvolutionNormalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionNormalLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size - 1) / 2), 
                              stride=1, dilation=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, z):
        real_part = self.conv(z.real)
        imag_part = self.conv(z.imag)

        bn_real = self.bn(real_part)
        bn_imag = self.bn(imag_part)

        return torch.complex(bn_real, bn_imag)
    

class ConvolutionConjLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvolutionConjLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=int((kernel_size - 1) / 2), 
                              stride=1, dilation=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, h):
        real_part = self.conv(h.real)
        imag_part = self.conv(h.imag)

        bn_real = self.bn(real_part)
        bn_imag = self.bn(imag_part)

        return torch.complex(bn_real, bn_imag)


class NonLinearLayer(nn.Module):
    def __init__(self, channels, breakpoints):
        super(NonLinearLayer, self).__init__()
        self.pwl = torchpwl.PWL(num_channels=channels, num_breakpoints=breakpoints)

    def forward(self, z):
        real_part = self.pwl(z.real)
        imag_part = self.pwl(z.imag)

        return torch.complex(real_part, imag_part)


class AdditionalLayer(nn.Module):
    def __init__(self, miu_1, miu_2, is_first=False):
        super(AdditionalLayer, self).__init__()
        self.miu_1 = miu_1
        self.miu_2 = miu_2
        self.is_first = is_first
    
    def forward(self, z, c, x, beta):
        variables = torch.add(x, beta)

        if self.is_first:
            return variables
        else:
            mid_value = torch.add(self.miu_1 * z, self.miu_2 * variables)
            return torch.sub(mid_value, c)
