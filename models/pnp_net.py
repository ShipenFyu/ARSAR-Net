import torch
import torchpwl
import torch.nn as nn

from utils.config import device_index

device = torch.device(device_index if torch.cuda.is_available() else "cpu")


class ADMMPnPNet(nn.Module):
    def __init__(
        self, 
        ob_matrix_left,
        ob_matrix_right,
        operator,
        num_phase,
        iteration,
        regular,
        in_channels = 1,
        out_channels = 32,
        kernel_size = 3, 
        num_breakpoints = 60,
        ):
        super(ADMMPnPNet, self).__init__()
        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.eta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        self.reconstruction_start = ReconstructionLayer(self.rho, ob_matrix_right, 
                                                        ob_matrix_left, operator, is_first=True)
        self.reconstruction_end = ReconstructionLayer(self.rho, ob_matrix_right, 
                                                      ob_matrix_left, operator, is_first=False)
        self.multiple = MultipleLayer(self.eta, is_first=True)
        layers = []

        for _ in range(num_phase):
            layers.append(BasicBlock(ob_matrix_right, ob_matrix_left, operator, regular, in_channels, out_channels, 
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
    def __init__(
            self, 
            ob_matrix_right, 
            ob_matrix_left, 
            operator, 
            regular, 
            in_channels, 
            out_channels, 
            kernel_size, 
            num_breakpoints, 
            iteration, 
            rho, 
            eta,
            ):
        super(BasicBlock, self).__init__()
        regularization = {'l1': SoftThresLayer, 'tv': TotalVarLayer, 'lr': RecurrentBlock}

        self.reconstruction = ReconstructionLayer(rho, ob_matrix_right, ob_matrix_left, operator)
        self.multiple = MultipleLayer(eta)

        if regular == 'l1':
            self.regular_layer = regularization[regular]()
        elif regular == 'tv':
            self.regular_layer = regularization[regular](iteration)
        elif regular == 'lr':
            self.regular_layer = regularization[regular](in_channels, out_channels, kernel_size, 
                                                         num_breakpoints, iteration)
        else:
            raise ValueError(f'unknown regularization {regular} found!')
    
    def forward(self, input_dict):
        input = input_dict['input']
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction(input, z, beta)
        z = self.regular_layer(x, beta)
        beta = self.multiple(beta, x, z)

        input_dict['x'] = x
        input_dict['z'] = z
        input_dict['beta'] = beta

        return input_dict


class ReconstructionLayer(nn.Module):
    def __init__(self, rho, Phi_fast, Phi_slow, operator, is_first=False):
        super(ReconstructionLayer, self).__init__()
        self.rho = rho
        self.operator = operator
        self.is_first = is_first

        Phi_fast_H = Phi_fast.conj().permute(0, 2, 1)
        Phi_slow_H = Phi_slow.conj().permute(0, 2, 1)
        Phi_fast_a = torch.matmul(Phi_fast, Phi_fast_H)
        Phi_slow_a = torch.matmul(Phi_slow_H, Phi_slow)

        rho_detached = self.rho.detach()  # it's ok?
        
        identity_right = rho_detached.to(device) * torch.eye(Phi_fast_a.shape[1]).expand(Phi_fast_a.shape[0], -1, -1).to(device)
        identity_left = rho_detached.to(device) * torch.eye(Phi_slow_a.shape[1]).expand(Phi_slow_a.shape[0], -1, -1).to(device)
        coffe_matrix_right = Phi_fast_a + identity_right
        coffe_matrix_left = Phi_slow_a + identity_left
        
        self.Phi_fast_H = Phi_fast_H
        self.Phi_slow_H = Phi_slow_H
        self.inverse_matrix_right = torch.inverse(coffe_matrix_right)
        self.inverse_matrix_left = torch.inverse(coffe_matrix_left)        

    def forward(self, input, z, beta):
        trivial_value = torch.matmul(torch.matmul(self.Phi_slow_H, input), self.Phi_fast_H)
        if self.is_first:
            value = torch.zeros_like(trivial_value, device=device_index)  # initialize
        else:
            value = torch.sub(z, beta)
        mid_value = trivial_value * self.operator + self.rho * value

        return torch.matmul(torch.matmul(self.inverse_matrix_left, mid_value), self.inverse_matrix_right)


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
    

class SoftThresLayer(nn.Module):
    def __init__(self):
        super(SoftThresLayer, self).__init__()
        self.soft_thres = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

    def forward(self, x, beta):
        '''
        Cited from: "Robust and Efficient Sparse-feature Enhancement 
        for Generalized SAR Imagery"
        '''
        residual = torch.add(x, beta)
        amplitude = torch.abs(residual)

        scale = amplitude - self.soft_thres
        z = torch.mul(torch.div(residual, amplitude), torch.where(scale > 0, scale, 0))

        return z
    

class TotalVarLayer(nn.Module):
    def __init__(self, iteration):
        super(TotalVarLayer, self).__init__()
        self.iteration = iteration

        self.miu_1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.miu_2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.grad_coffe = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

    def tv_gradient(self, z):
        width_dif = torch.zeros_like(z)
        height_dif = torch.zeros_like(z)

        width_dif_s = torch.zeros_like(z)
        height_dif_s = torch.zeros_like(z)

        # wrap-around
        width_dif[:, :, :-1] = z[:, :, 1:] - z[:, :, :-1]
        width_dif[:, :, -1] = z[:, :, 0] - z[:, :, -1]

        height_dif[:, :-1, :] = z[:, 1:, :] - z[:, :-1, :]
        height_dif[:, -1, :] = z[:, 0, :] - z[:, -1, :]

        # second order differential
        width_dif_s[:, :, :-1] = width_dif[:, :, 1:] - width_dif[:, :, :-1]
        width_dif_s[:, :, -1] = width_dif[:, :, 0] - width_dif[:, :, -1]

        height_dif_s[:, :-1, :] = height_dif[:, 1:, :] - height_dif[:, :-1, :]
        height_dif_s[:, -1, :] = height_dif[:, 0, :] - height_dif[:, -1, :]

        grad = torch.zeros_like(z)
        grad = 2 * (width_dif_s + height_dif_s)

        return grad
    
    def additional_update(self, z, residual, grad):
        mid_value = torch.add(self.miu_1 * z, self.miu_2 * residual)

        return torch.sub(mid_value, self.grad_coffe * grad)
    
    def forward(self, x, beta):
        residual = torch.add(x, beta)
        z = residual

        for _ in range(self.iteration):
            grad = self.tv_gradient(z)
            z = self.additional_update(z, residual, grad)

        return z


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
    