import torch
import torchpwl
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torch.fft import fft, ifft, fftshift, ifftshift


class ARSARNet(nn.Module):
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
        super(ARSARNet, self).__init__()
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
            layers.append(BasicBlock(device_index, operator, up_matrix, regular, in_channels, out_channels, 
                                     base_channels, kernel_size, num_breakpoints, iteration, self.rho, self.eta))
        
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
        regularization = {'cnn': RecurrentBlock, 'unet': UnetUpdateLayer, 
                          'pretrain': UnetWithPretrain}

        self.reconstruction = ReconstructionLayer(rho, operator, up_matrix, device_index)
        self.multiple = MultipleLayer(eta, device_index)

        if regular == 'cnn':
            self.regular_layer = regularization[regular](in_channels, out_channels, kernel_size, 
                                                         num_breakpoints, iteration)
        elif regular == 'unet':
            self.regular_layer = regularization[regular](in_channels, base_channels, kernel_size, 
                                                         iteration)
        elif regular == 'pretrain':
            self.regular_layer = regularization[regular](in_channels, kernel_size)
        else:
            raise ValueError(f'unknown regularization {regular} found!')
    
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
        rec_echo = torch.einsum('ij,bjk->bik', self.up_matrix, echo)
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


class UnetWithPretrain(nn.Module):
    '''
    Backbone: ResNet50 with pretrain
    Parameters: 158M(158,713,924)
    '''
    def __init__(self, in_channels, kernel_size):
        super(UnetWithPretrain, self).__init__()
        resnet = resnet18(pretrained=True)
        
        self.initial = nn.Conv2d(in_channels, 3, kernel_size=1)

        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2  # out_channels = 128
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4  # out_channels = 512

        self.center = ConvLayer(512, 512, kernel_size)

        self.decoder5 = ConvLayer(1024, 256, kernel_size)
        self.decoder4 = ConvLayer(512, 128, kernel_size)
        self.decoder3 = ConvLayer(256, 64, kernel_size)
        self.decoder2 = ConvLayer(128, 32, kernel_size)
        self.decoder1 = ConvLayer(32 + 64, 16, kernel_size)
        self.reshape = ConvLayer(16 + 3, 16, kernel_size)

        self.out = nn.Conv2d(16, in_channels, kernel_size=1)

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
    
    def layer_forwrd(self, layer: nn.Module, z):
        real_z = layer(z.real)
        imag_z = layer(z.imag)

        z_out = torch.complex(real_z, imag_z)

        return z_out

    def forward(self, x, beta):
        z = torch.add(x, beta)
        z_channeled = torch.unsqueeze(z, 1)

        initial = self.layer_forwrd(self.initial, z_channeled)  # channel = 3, size = 512
        enc1 = self.layer_forwrd(self.encoder1, initial)  # channel = 64, size = 256
        enc2 = self.layer_forwrd(self.encoder2, enc1)
        enc2 = self.average_pooling(enc2, 2)  # channel = 64, size = 128
        enc3 = self.layer_forwrd(self.encoder3, enc2)  # channel = 128, size = 64
        enc4 = self.layer_forwrd(self.encoder4, enc3)  # channel = 256, size = 32
        enc5 = self.layer_forwrd(self.encoder5, enc4)  # channel = 512, size = 16

        center = self.center(self.average_pooling(enc5, 2))  # channel = 512, size = 8

        dec5 = self.decoder5(torch.cat([self.interpolate(center, scale_factor=2), enc5], 1))  # channel: 1024->256, size = 16
        dec4 = self.decoder4(torch.cat([self.interpolate(dec5, scale_factor=2), enc4], 1))  # channel: 512->128, size = 32
        dec3 = self.decoder3(torch.cat([self.interpolate(dec4, scale_factor=2), enc3], 1))  # channel: 256->64, size = 64
        dec2 = self.decoder2(torch.cat([self.interpolate(dec3, scale_factor=2), enc2], 1))  # channel: 128->32, size = 128
        dec1 = self.decoder1(torch.cat([self.interpolate(dec2, scale_factor=2), enc1], 1))  # channel: 64+32->16, size = 256

        end = self.reshape(torch.cat([self.interpolate(dec1, scale_factor=2), initial], 1))  # channel: 16+3->16, size = 512

        output = self.layer_forwrd(self.out, end)
        result = torch.squeeze(output, 1)

        return result
    

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
    

class UNetWithFPN(nn.Module):
    def __init__(self, backbone=resnet18(pretrained=True)):
        super(UNetWithFPN, self).__init__()
        # Backbone (ResNet)
        layer_list = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.encoders = [
            layer_list.layer1,  # C2
            layer_list.layer2,  # C3
            layer_list.layer3,  # C4
            layer_list.layer4,  # C5
        ]
        
        # Lateral connections for FPN
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.Conv2d(2048, 256, kernel_size=1),
        ])
        
        # Decoder for U-Net
        self.decoders = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        ])
        
        # Output layer
        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)
    
    def forward(self, x):
        # Bottom-up: ResNet features
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        
        # Top-down: FPN
        p5 = self.lateral_convs[3](c5)
        p4 = self.lateral_convs[2](c4) + nn.functional.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral_convs[1](c3) + nn.functional.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral_convs[0](c2) + nn.functional.interpolate(p3, scale_factor=2, mode='nearest')
        
        # Decoder (U-Net like)
        d1 = self.decoders[0](p2)
        d2 = self.decoders[1](nn.functional.interpolate(d1, scale_factor=2, mode='nearest'))
        d3 = self.decoders[2](nn.functional.interpolate(d2, scale_factor=2, mode='nearest'))
        
        # Final output
        output = self.output_conv(nn.functional.interpolate(d3, size=x.shape[-2:], mode='bilinear', align_corners=False))
        return output
