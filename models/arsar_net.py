import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.fft import fft, ifft, fftshift, ifftshift


class ARSARNet(nn.Module):
    def __init__(
        self, 
        device, 
        operator,
        down_matrix, 
        up_matrix, 
        layer_num,
        regular,
        in_channels = 1,
        base_channels = 4,
        kernel_size = 3, 
        ):
        super(ARSARNet, self).__init__()
        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.eta = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.device = device

        self.reconstruction_start = ReconstructionLayer(self.rho, operator, down_matrix, 
                                                        up_matrix, device, is_first=True)
        self.reconstruction_end = ReconstructionLayer(self.rho, operator, down_matrix, 
                                                        up_matrix, device, is_first=False)
        self.multiple = MultipleLayer(self.eta, is_first=True)
        layers = []

        for _ in range(layer_num):
            layers.append(BasicBlock(device, operator, down_matrix, up_matrix, regular, in_channels,  
                                     base_channels, kernel_size, self.rho, self.eta))
        
        self.iteration_net = nn.Sequential(*layers)
    
    def forward(self, echo):
        x = self.reconstruction_start(echo, 0, 0, 0)
        beta = self.multiple(0, x, 0)
        z = torch.zeros_like(x, device=x.device)

        input_dict = dict()
        input_dict['echo'] = echo
        input_dict['x'] = x
        input_dict['z'] = z
        input_dict['beta'] = beta

        input_dict = self.iteration_net(input_dict)
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction_end(echo, x, z, beta)

        return x


class BasicBlock(nn.Module):
    def __init__(
            self, 
            device, 
            operator,
            down_matrix,  
            up_matrix, 
            regular, 
            in_channels, 
            base_channels, 
            kernel_size, 
            rho, 
            eta,
            ):
        super(BasicBlock, self).__init__()
        regularization = {'swift': SwiftNet, 'pro': ProNet}
        swift_channels = base_channels
        pro_channels = base_channels * 4

        self.reconstruction = ReconstructionLayer(rho, operator, down_matrix, up_matrix, device)
        self.multiple = MultipleLayer(eta)

        if regular == 'swift':
            self.regular_layer = regularization[regular](in_channels, swift_channels, kernel_size)
        elif regular == 'pro':
            self.regular_layer = regularization[regular](in_channels, pro_channels, kernel_size)
        else:
            raise ValueError(f'Unknown regularization found: {regular}!')
    
    def forward(self, input_dict):
        echo = input_dict['echo']
        x = input_dict['x']
        z = input_dict['z']
        beta = input_dict['beta']

        x = self.reconstruction(echo, x, z, beta)
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
    def __init__(self, rho, operator, down_matrix, up_matrix, device, is_first=False):
        super(ReconstructionLayer, self).__init__()
        self.rho = rho
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.operator = operator
        self.is_first = is_first
        self.down_matrix = down_matrix
        self.up_matrix = up_matrix
        self.device = device

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
    
    def echo_operator(self, image):
        '''
        Azimuth FFT -> Conjugate Azimuth Compression -> Range FFT 
        -> Conjugate Range compression -> Range IFFT -> Conjugate RCMC -> Azimuth IFFT
        '''
        image_fa = fftshift(fft(image, dim=1, norm='ortho'), dim=1)
        image_fa = image_fa * torch.conj(self.operator['ac']).to(self.device)
        image_far = fftshift(fft(image_fa, dim=2, norm='ortho'), dim=2)
        image_far = image_far * torch.conj(self.operator['rc']).to(self.device)
        image_fa = ifft(ifftshift(image_far, dim=2), dim=2, norm='ortho')
        image_fa = image_fa * torch.conj(self.operator['sc']).to(self.device)
        echo = ifft(ifftshift(image_fa, dim=1), dim=1, norm='ortho')
        shaped_echo = torch.einsum('ij,bjk->bik', self.down_matrix, echo)

        return shaped_echo

    def forward(self, echo_input, x, z, beta):
        if self.is_first:
            trivial_value = self.imaging_operator(echo_input)
            addition_value = torch.zeros_like(trivial_value, device=echo_input.device)  # initialize
        else:
            residual = echo_input - self.echo_operator(x)
            trivial_value = self.imaging_operator(residual)
            penalty_value = self.rho * torch.sub(z, beta)
            scaled_value = (1 - self.rho) * x
            addition_value = scaled_value + penalty_value

        return self.gamma * trivial_value + addition_value


class MultipleLayer(nn.Module):
    def __init__(self, eta, is_first=False):
        super(MultipleLayer, self).__init__()
        self.eta = eta
        self.is_first = is_first

    def forward(self, beta, x, z):
        if self.is_first:
            return torch.zeros_like(x, device=x.device)  # initialize
        else:
            return torch.add(beta, self.eta * torch.sub(x, z))


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvLayer, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                padding=int((kernel_size - 1) / 2))
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ELU()

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                                padding=int((kernel_size - 1) / 2))
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ELU()

    def forward(self, z):
        conv1_r = self.conv_1(z.real)
        conv1_i = self.conv_1(z.imag)
        bn1_r = self.bn_1(conv1_r)
        bn1_i = self.bn_1(conv1_i)
        relu1_r = self.relu_1(bn1_r)
        relu1_i = self.relu_1(bn1_i)

        conv2_r = self.conv_2(relu1_r)
        conv2_i = self.conv_2(relu1_i)
        bn2_r = self.bn_2(conv2_r)
        bn2_i = self.bn_2(conv2_i)
        relu2_r = self.relu_2(bn2_r)
        relu2_i = self.relu_2(bn2_i)

        return torch.complex(relu2_r, relu2_i)


class SwiftNet(nn.Module):
    '''
    Network for z updating with high inference speed
    Parameters: 1.3M(1,317,334)
    '''
    def __init__(self, in_channels, base_channels, kernel_size):
        super(SwiftNet, self).__init__()
        self.encoder1 = ConvLayer(in_channels, base_channels, kernel_size)
        self.encoder2 = ConvLayer(base_channels, base_channels * 4, kernel_size)
        self.encoder3 = ConvLayer(base_channels * 4, base_channels * 16, kernel_size)

        self.center = ConvLayer(base_channels * 16, base_channels * 16, kernel_size)

        self.decoder3 = ConvLayer(base_channels * 32, base_channels * 4, kernel_size)
        self.decoder2 = ConvLayer(base_channels * 8, base_channels, kernel_size)
        self.decoder1 = ConvLayer(base_channels * 2, base_channels, kernel_size)

        self.out = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def average_pooling(self, encoder, kernel_size):
        encoder_r = F.avg_pool2d(encoder.real, kernel_size)
        encoder_i = F.avg_pool2d(encoder.imag, kernel_size)

        return torch.complex(encoder_r, encoder_i)
    
    def interpolate(self, decoder, scale_factor):
        decoder_r = F.interpolate(decoder.real, scale_factor=scale_factor, 
                                     mode='bilinear', align_corners=True)
        decoder_i = F.interpolate(decoder.imag, scale_factor=scale_factor, 
                                     mode='bilinear', align_corners=True)
        
        return torch.complex(decoder_r, decoder_i)

    def forward(self, x, beta):
        z = torch.add(x, beta)
        z_channeled = torch.unsqueeze(z, 1)

        enc1 = self.encoder1(z_channeled)  # channel = 8, size = 512
        enc2 = self.encoder2(self.average_pooling(enc1, 4))  # channel = 32, size = 128
        enc3 = self.encoder3(self.average_pooling(enc2, 4))  # channel = 128, size = 32

        center = self.center(self.average_pooling(enc3, 2))  # channel = 128, size = 16

        dec3 = self.decoder3(torch.cat([self.interpolate(center, scale_factor=2), enc3], 1))
        dec2 = self.decoder2(torch.cat([self.interpolate(dec3, scale_factor=4), enc2], 1))
        dec1 = self.decoder1(torch.cat([self.interpolate(dec2, scale_factor=4), enc1], 1))
        output_r = self.out(dec1.real)
        output_i = self.out(dec1.imag)
        output = torch.complex(output_r, output_i)

        result = torch.squeeze(output, 1)

        return result


class ProNet(nn.Module):
    '''
    Network of precise reconstruction for z updating
    Parameters: 1.5M(1,500,502)
    '''
    def __init__(self, in_channels, base_channels, kernel_size):
        super(ProNet, self).__init__()
        padding = int((kernel_size - 1) / 2)

        self.encoder_i = nn.Conv2d(in_channels, base_channels, kernel_size, padding=padding)
        self.encoder_1 = nn.Conv2d(base_channels, base_channels * 4, kernel_size, padding=padding)
        self.bn_1 = nn.BatchNorm2d(base_channels * 4)
        self.relu_1 = nn.ELU()
        self.encoder_2 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size, padding=padding)

        self.threshold = nn.ELU()

        self.decoder_2 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size, padding=padding)
        self.bn_2 = nn.BatchNorm2d(base_channels * 4)
        self.relu_2 = nn.ELU()
        self.decoder_1 = nn.Conv2d(base_channels * 4, base_channels, kernel_size, padding=padding)
        self.decoder_e = nn.Conv2d(base_channels, in_channels, kernel_size, padding=padding)

    def layer_forwrd(self, layer: nn.Module, feature):
        feature_r = layer(feature.real)
        feature_i = layer(feature.imag)

        return torch.complex(feature_r, feature_i)

    def forward(self, x, beta):
        z = torch.add(x, beta)
        z_channeled = torch.unsqueeze(z, 1)

        enci = self.layer_forwrd(self.encoder_i, z_channeled)
        enc1 = self.layer_forwrd(self.encoder_1, enci)
        bn1 = self.layer_forwrd(self.bn_1, enc1)
        relu1 = self.layer_forwrd(self.relu_1, bn1)
        enc2 = self.layer_forwrd(self.encoder_2, relu1)

        relu_c = self.layer_forwrd(self.threshold, enc2)

        dec2 = self.layer_forwrd(self.decoder_2, relu_c)
        bn2 = self.layer_forwrd(self.bn_2, dec2)
        relu2 = self.layer_forwrd(self.relu_2, bn2)
        dec1 = self.layer_forwrd(self.decoder_1, relu2)

        dece = self.layer_forwrd(self.decoder_e, dec1) + z_channeled
        output = torch.squeeze(dece, 1)

        return output
