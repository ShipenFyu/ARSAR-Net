import torch
import torchpwl
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch.fft import fft, ifft, fftshift, ifftshift
from pytorch_wavelets import DWTForward


class NonInversionADMMPnPNet(nn.Module):
    def __init__(
        self, 
        device_index, 
        operator,
        up_matrix, 
        num_phase,
        iteration,
        regular,
        in_channels = 1,
        out_channels = 32,
        base_channels = 8,  # over fitting
        kernel_size = 3, 
        num_breakpoints = 60,
        ):
        super(NonInversionADMMPnPNet, self).__init__()
        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.eta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.device_index = device_index

        self.reconstruction_start = ReconstructionLayer(self.rho, operator, 
                                                        up_matrix, device_index, is_first=True)
        self.reconstruction_end = ReconstructionLayer(self.rho, operator, 
                                                        up_matrix, device_index, is_first=False)
        self.multiple = MultipleLayer(self.eta, device_index, is_first=True)
        layers = []

        for _ in range(num_phase):
            layers.append(BasicBlock(device_index, operator, up_matrix, regular, in_channels, out_channels, base_channels, 
                                     kernel_size, num_breakpoints, iteration, self.rho, self.eta))
        
        self.iteration_net = nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.reconstruction_start(input, 0, 0, 0)
        beta = self.multiple(0, x, 0)
        z = torch.zeros_like(x, device=self.device_index)

        input_dict = dict()
        input_dict['input'] = input
        input_dict['x'] = x
        input_dict['z'] = z
        input_dict['beta'] = beta

        input_dict = self.iteration_net(input_dict)
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction_end(input, x, z, beta)

        return x


class BasicBlock(nn.Module):
    def __init__(
            self, 
            device_index, 
            operator, 
            up_matrix, 
            regular, 
            in_channels, 
            out_channels,
            base_channels, 
            kernel_size, 
            num_breakpoints, 
            iteration, 
            rho, 
            eta,
            ):
        super(BasicBlock, self).__init__()
        regularization = {'l1': SoftThresLayer, 'tv': TotalVarLayer, 
                          'ir': RecurrentBlock, 'unet': UnetUpdateLayer,
                          'haar': ScaleFusionLayer}

        self.reconstruction = ReconstructionLayer(rho, operator, up_matrix, device_index)
        self.multiple = MultipleLayer(eta, device_index)

        if regular == 'l1':
            self.regular_layer = regularization[regular]()
        elif regular == 'tv':
            self.regular_layer = regularization[regular](iteration)
        elif regular == 'ir':
            self.regular_layer = regularization[regular](in_channels, out_channels, kernel_size, 
                                                         num_breakpoints, iteration)
        elif regular == 'unet':
            self.regular_layer = regularization[regular](in_channels, base_channels, kernel_size, 
                                                         iteration)
        elif regular == 'haar':
            self.regular_layer = regularization[regular](in_channels, kernel_size, device_index)
        else:
            raise ValueError(f'Unknown regularization found: {regular}!')
    
    def forward(self, input_dict):
        input = input_dict['input']
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction(input, x, z, beta)
        z = self.regular_layer(x, beta)
        beta = self.multiple(beta, x, z)

        input_dict['x'] = x
        input_dict['z'] = z
        input_dict['beta'] = beta

        return input_dict


class ReconstructionLayer(nn.Module):
    '''
    Non inversion ADMM: Optimization using second-order Taylor expansion, replacing matrix inversion step
    Method cited from: A 3-D Sparse SAR Imaging Method Based on Plug-and-Play
    '''
    def __init__(self, rho, operator, up_matrix, device_index, is_first=False):
        super(ReconstructionLayer, self).__init__()
        self.rho = rho
        self.gamma = nn.Parameter(torch.tensor([-0.1]), requires_grad=True)
        self.operator = operator
        self.is_first = is_first
        self.up_matrix = up_matrix
        self.device_index = device_index
        self.device = torch.device(device_index if torch.cuda.is_available() else "cpu")

    def imaging_operator(self, echo):
        '''
        Nonuniform Azimuth FFT -> RCMC -> Range FFT -> Range Compression
        -> Range IFFT -> Azimuth Compression -> Azimuth IFFT 
        '''
        rec_echo = torch.matmul(self.up_matrix, echo)
        echo_fa = fftshift(fft(rec_echo, dim=1, norm='ortho'), dim=1)
        echo_fa = echo_fa * self.operator['sc'].to(self.device)
        echo_far = fftshift(fft(echo_fa, dim=2, norm='ortho'), dim=2)
        echo_far = echo_far * self.operator['rc'].to(self.device)
        echo_fa = ifft(ifftshift(echo_far, dim=2), dim=2, norm='ortho')
        echo_fa = echo_fa * self.operator['ac'].to(self.device)
        image = ifft(ifftshift(echo_fa, dim=1), dim=1, norm='ortho')
        
        return image

    def forward(self, input, x, z, beta):
        trivial_value = self.imaging_operator(input)
        if self.is_first:
            addition_value = torch.zeros_like(trivial_value, device=self.device_index)  # initialize
        else:
            penalty_value = self.rho * torch.sub(z, beta)
            scaled_value = self.gamma * x
            addition_value = scaled_value + penalty_value

        return trivial_value + addition_value


class MultipleLayer(nn.Module):
    def __init__(self, eta, device_index, is_first=False):
        super(MultipleLayer, self).__init__()
        self.eta = eta
        self.device_index = device_index
        self.is_first = is_first

    def forward(self, beta, x, z):
        if self.is_first:
            return torch.zeros_like(x, device=self.device_index)  # initialize
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


class UnetUpdateLayer(nn.Module):
    '''
    Unet structure for z updating
    '''
    def __init__(self, in_channels, base_channels, kernel_size, iteration):
        super(UnetUpdateLayer, self).__init__()
    
        self.iteration = iteration

        self.miu_1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.miu_2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.unet = UnetLayer(in_channels, base_channels, kernel_size)
        self.additional_start = AdditionalLayer(self.miu_1, self.miu_2, is_first=True)
        self.additional = AdditionalLayer(self.miu_1, self.miu_2)

    def forward(self, x, beta):
        z = self.additional_start(0, 0, x, beta)  # initialize

        for _ in range(self.iteration):
            z_channeled = torch.unsqueeze(z, 1)
            unet_out = self.unet(z_channeled)
            unet_shaped = torch.squeeze(unet_out, 1)
            z = self.additional(z, unet_shaped, x, beta)
        
        return z

class UnetLayer(nn.Module):
    def __init__(self, in_channels, base_channels, kernel_size):
        super(UnetLayer, self).__init__()
        self.encoder1 = ConvLayer(in_channels, base_channels, kernel_size)
        self.encoder2 = ConvLayer(base_channels, base_channels * 2, kernel_size)
        self.encoder3 = ConvLayer(base_channels * 2, base_channels * 4, kernel_size)

        self.center = ConvLayer(base_channels * 4, base_channels * 4, kernel_size)

        self.decoder3 = ConvLayer(base_channels * 8, base_channels * 2, kernel_size)
        self.decoder2 = ConvLayer(base_channels * 4, base_channels, kernel_size)
        self.decoder1 = ConvLayer(base_channels * 2, base_channels, kernel_size)

        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def average_pooling(self, encoder, kernel_size=2):
        encoder_real = F.avg_pool2d(encoder.real, kernel_size)
        encoder_imag = F.avg_pool2d(encoder.imag, kernel_size)

        return torch.complex(encoder_real, encoder_imag)
    
    def interpolate(self, decoder, scale_factor):
        decoder_real = F.interpolate(decoder.real, scale_factor=scale_factor, 
                                     mode='bilinear', align_corners=True)
        decoder_imag = F.interpolate(decoder.imag, scale_factor=scale_factor, 
                                     mode='bilinear', align_corners=True)
        
        return torch.complex(decoder_real, decoder_imag)

    def forward(self, z):
        enc1 = self.encoder1(z)
        enc2 = self.encoder2(self.average_pooling(enc1, 2))
        enc3 = self.encoder3(self.average_pooling(enc2, 2))

        center = self.center(self.average_pooling(enc3, 2))

        dec3 = self.decoder3(torch.cat([self.interpolate(center, scale_factor=2), enc3], 1))
        dec2 = self.decoder2(torch.cat([self.interpolate(dec3, scale_factor=2), enc2], 1))
        dec1 = self.decoder1(torch.cat([self.interpolate(dec2, scale_factor=2), enc1], 1))
        output_real = self.out(dec1.real)
        output_imag = self.out(dec1.imag)

        return torch.complex(output_real, output_imag)
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                               padding=int((kernel_size - 1) / 2))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                               padding=int((kernel_size - 1) / 2))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, z):
        real_conv1 = self.conv1(z.real)
        imag_conv1 = self.conv1(z.imag)
        real_bn1 = self.bn1(real_conv1)
        imag_bn1 = self.bn1(imag_conv1)
        real_relu1 = self.relu1(real_bn1)
        imag_relu1 = self.relu1(imag_bn1)

        real_conv2 = self.conv2(real_relu1)
        imag_conv2 = self.conv2(imag_relu1)
        real_bn2 = self.bn2(real_conv2)
        imag_bn2 = self.bn2(imag_conv2)
        real_relu2 = self.relu2(real_bn2)
        imag_relu2 = self.relu2(imag_bn2)

        return torch.complex(real_relu2, imag_relu2)


class ScaleFusionLayer(nn.Module):
    '''
    Referred to "RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation"
    Backbone: ResNet50 with pretrain, parameters: 1.7B(1,715,560,384)
              ResNet34 with pretrain, parameters: 319M(319,373,365)
    '''
    def __init__(self, in_channels, kernel_size, device_index, net='resnet34'):
        super(ScaleFusionLayer, self).__init__()
        self.device = torch.device(device_index if torch.cuda.is_available() else "cpu")
        if net == 'resnet50':
            resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            channel_list=[3, 64, 256, 512, 1024, 2048]
        elif net == 'resnet34':
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
            channel_list=[3, 64, 64, 128, 256, 512]
        elif net == 'resnet18':
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            channel_list=[3, 64, 64, 128, 256, 512]
        else:
            raise ValueError(f'Unknown ResNet type {net} found!')
        
        self.initial = nn.Conv2d(in_channels, 3, kernel_size=1)

        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )  # in_channels = 3, out_channels = 64 for resnet50
        self.encoder2 = nn.Sequential(
            resnet.maxpool, 
            resnet.layer1, 
        )  # in_channels = 64, out_channels = 256  
        self.encoder3 = resnet.layer2  # in_channels = 256, out_channels = 512
        self.encoder4 = resnet.layer3  # in_channels = 512, out_channels = 1024
        self.encoder5 = resnet.layer4  # in_channels = 1024, out_channels = 2048

        self.rcu0 = ResidualUnit(channel_list[0], kernel_size)
        self.rcu1 = ResidualUnit(channel_list[1], kernel_size)
        self.rcu2 = ResidualUnit(channel_list[2], kernel_size)
        self.rcu3 = ResidualUnit(channel_list[3], kernel_size)
        self.rcu4 = ResidualUnit(channel_list[4], kernel_size)
        self.rcu5 = ResidualUnit(channel_list[5], kernel_size)

        self.fb4 = FusionBlock(channel_list[4], channel_list[5], kernel_size, self.device)
        self.fb3 = FusionBlock(channel_list[3], channel_list[4], kernel_size, self.device)
        self.fb2 = FusionBlock(channel_list[2], channel_list[3], kernel_size, self.device)
        self.fb1 = FusionBlock(channel_list[1], channel_list[2], kernel_size, self.device)
        self.fb0 = FusionBlock(channel_list[0], channel_list[1], kernel_size, self.device)

        self.out = nn.Sequential(
            nn.Conv2d(channel_list[0], channel_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_list[0], in_channels, kernel_size=1, bias=False),
        )

    def layer_forwrd(self, layer: nn.Module, z):
        real_z = layer(z.real)
        imag_z = layer(z.imag)

        return torch.complex(real_z, imag_z)
    
    def forward(self, x, beta):
        z = torch.add(x, beta)
        z_channeled = torch.unsqueeze(z, 1)

        initial = self.layer_forwrd(self.initial, z_channeled)  # channel = 3, size = 512 for resnet50
        enc1 = self.layer_forwrd(self.encoder1, initial)  # channel = 64, size = 256
        enc2 = self.layer_forwrd(self.encoder2, enc1)  # channel = 256, size = 128
        enc3 = self.layer_forwrd(self.encoder3, enc2)  # channel = 512, size = 64
        enc4 = self.layer_forwrd(self.encoder4, enc3)  # channel = 1024, size = 32
        enc5 = self.layer_forwrd(self.encoder5, enc4)  # channel = 2048, size = 16

        ru0 = self.rcu0(initial)
        ru1 = self.rcu1(enc1)
        ru2 = self.rcu2(enc2)
        ru3 = self.rcu3(enc3)
        ru4 = self.rcu4(enc4)
        ru5 = self.rcu5(enc5)

        fusion4 = self.fb4(ru4, ru5)  # channel: 1024, 2048->1024, size = 32
        fusion3 = self.fb3(ru3, fusion4)  # channel: 512, 1024->512, size = 64
        fusion2 = self.fb2(ru2, fusion3)  # channel: 256, 512->256, size = 128
        fusion1 = self.fb1(ru1, fusion2)  # channel: 64, 256->256, size = 256
        fusion0 = self.fb0(ru0, fusion1)  # channel: 3, 64->3, size = 512

        output = self.layer_forwrd(self.out, fusion0)
        result = torch.squeeze(output, 1)

        return result


class ResidualUnit(nn.Module):
    '''
    Residual Convolution Unit cited from: 
    "RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation"
    '''
    def __init__(self, in_channels, kernel_size):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, 
                               padding=int((kernel_size - 1) / 2), bias=False)  # keep data consistent, bias = False
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, 
                               padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=False)
    
    def forward(self, z):
        real_conv1 = self.conv1(z.real)
        imag_conv1 = self.conv1(z.imag)
        real_bn1 = self.bn1(real_conv1)
        imag_bn1 = self.bn1(imag_conv1)
        real_relu = self.relu1(real_bn1)
        imag_relu = self.relu1(imag_bn1)

        real_conv2 = self.conv2(real_relu)
        imag_conv2 = self.conv2(imag_relu)
        real_bn2 = self.bn2(real_conv2)
        imag_bn2 = self.bn2(imag_conv2)
        real_relu2 = self.relu2(real_bn2)
        imag_relu2 = self.relu2(imag_bn2)

        return torch.complex(real_relu2, imag_relu2) + z


class FusionBlock(nn.Module):
    '''
    Multi-resolution feature map fusion block cited from: 
    "RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation"
    '''
    def __init__(self, in_channels, scaled_channels, kernel_size, device, scale_factor=2):
        super(FusionBlock, self).__init__()
        self.scale_factor = scale_factor
        self.haar = ChainedHaarSampling(in_channels, kernel_size, device)
        self.conv_l = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                padding=int((kernel_size - 1) / 2), bias=False)
        self.conv_s = nn.Conv2d(scaled_channels, scaled_channels, kernel_size, 
                                padding=int((kernel_size - 1) / 2), bias=False)
        self.conv = nn.Conv2d(in_channels * 5 + scaled_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, feature, feature_s):
        haar_feature = self.haar(feature)  # size -> size/2, channels = in_channels * 4 
        real_feature = self.conv_l(feature.real)
        imag_feature = self.conv_l(feature.imag)

        real_feature_s = self.conv_s(feature_s.real)
        imag_feature_s = self.conv_s(feature_s.imag)

        real_feature_s = torch.cat([real_feature_s, haar_feature.real], 1) 
        imag_feature_s = torch.cat([imag_feature_s, haar_feature.imag], 1)
        real_interp = F.interpolate(real_feature_s, scale_factor=self.scale_factor, 
                                    mode='bilinear', align_corners=True)
        imag_interp = F.interpolate(imag_feature_s, scale_factor=self.scale_factor, 
                                    mode='bilinear', align_corners=True)
        
        real_concat = torch.cat([real_interp, real_feature], 1)
        imag_concat = torch.cat([imag_interp, imag_feature], 1)

        real_conv = self.conv(real_concat)
        imag_conv = self.conv(imag_concat)
        real_bn = self.bn(real_conv)
        imag_bn = self.bn(imag_conv)
        real_relu = self.relu(real_bn)
        imag_relu = self.relu(imag_bn)

        return torch.complex(real_relu, imag_relu)


class ChainedHaarSampling(nn.Module):
    def __init__(self, in_channels, kernel_size, device):
        super(ChainedHaarSampling, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(in_channels, in_channels, 
                               kernel_size, padding=int((kernel_size - 1) / 2))
        self.conv2 = nn.Conv2d(in_channels, in_channels, 
                               kernel_size, padding=int((kernel_size - 1) / 2))
        self.conv3 = nn.Conv2d(in_channels, in_channels, 
                               kernel_size, padding=int((kernel_size - 1) / 2))
        self.conv4 = nn.Conv2d(in_channels, in_channels, 
                               kernel_size, padding=int((kernel_size - 1) / 2))
        
    def dwt_downsampling(self, feature):
        dwt = DWTForward(J=1, mode='zero', wave='haar').to(self.device)
        coeffs_real = dwt(feature.real)
        coeffs_imag = dwt(feature.imag)
        ll_real, h_real = coeffs_real
        ll_imag, h_imag = coeffs_imag
        h_real, h_imag = h_real[0], h_imag[0]
        
        lh_real, hl_real, hh_real = h_real[:,:,0,:,:], h_real[:,:,1,:,:], h_real[:,:,2,:,:]
        lh_imag, hl_imag, hh_imag = h_imag[:,:,0,:,:], h_imag[:,:,1,:,:], h_imag[:,:,2,:,:]

        ll = torch.complex(ll_real, ll_imag)
        lh = torch.complex(lh_real, lh_imag)
        hl = torch.complex(hl_real, hl_imag)
        hh = torch.complex(hh_real, hh_imag)
        
        return ll, lh, hl, hh
    
    def layer_forwrd(self, layer: nn.Module, feature):
        real_feature = layer(feature.real)
        imag_feature = layer(feature.imag)

        return torch.complex(real_feature, imag_feature)
    
    def forward(self, feature):
        ll, lh, hl, hh = self.dwt_downsampling(feature)
        ll_conv = self.layer_forwrd(self.conv1, ll)  # in_channels
        lh_conv = self.layer_forwrd(self.conv2, lh)
        hl_conv = self.layer_forwrd(self.conv3, hl)
        hh_conv = self.layer_forwrd(self.conv4, hh)

        out = torch.cat([ll_conv, lh_conv, hl_conv, hh_conv], 1)  # in_channels * 4

        return out
