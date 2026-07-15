import os
import argparse
import warnings
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils.observation_matrix import random_sampling_create
from utils.evaluate import range_cut, nrmse_evaluate, psnr_evaluate, ssim_evaluate, aliasing_construct
from utils.config import processor
from models.arsar_net import ARSARNet
from models.pnp_net import NonInversionADMMPnPNet
from models.sr_net import SRNet


def get_args():
    parser = argparse.ArgumentParser(description='ARSAR-Net Testing')
    parser.add_argument('--test_image', required=True, help='Path to the test image .npy file')
    parser.add_argument('--test_echo', required=True, help='Path to the test echo .npy file')
    parser.add_argument('--weight', required=True, help='Path to the model checkpoint')
    parser.add_argument('--device', default='cuda:0', help='The device index used in testing')
    parser.add_argument('--network', default='arsar', help='Backbone network (sr, pnp or arsar)')
    parser.add_argument('--regularization', default='swift', help='ARSAR-Net variant (swift or pro)')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for testing')
    parser.add_argument('--layer_num', default=9, type=int, help='Net block num in iteration')
    parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
    parser.add_argument('--down_rate', default=0.5, type=float, help='Azimuth down-sampling rate')
    args = parser.parse_args()

    return args


def test_model(model: torch.nn.Module, test_loader: Iterable, 
         device: torch.device, batch_size: int, test_image
         ):
    rec = torch.zeros_like(test_image, dtype=torch.complex64).to(device)

    print('SAR Reconstruction started at', datetime.now().strftime("%H:%M:%S"))
    with torch.no_grad():
        for i, clip in enumerate(tqdm(test_loader, desc=f'Reconstruction')):
                _, echo = clip[0].to(device), clip[1].to(device)
                output = model(echo)
                rec[i * batch_size: (i + 1) * batch_size, :, :] = output
    print('Reconstruction completed at', datetime.now().strftime("%H:%M:%S"))

    return rec


def pre_process(rec, echo, img, up_matrix):
    output_dict = {}
    aliasing = aliasing_construct(echo, up_matrix, processor)

    rec = torch.abs(rec)
    echo = torch.abs(echo)
    aliasing = torch.abs(aliasing)
    img = torch.abs(img)

    # range cut
    rec_norm = range_cut(rec)
    echo_norm = range_cut(echo)
    aliasing_norm = range_cut(aliasing)
    img_norm = range_cut(img)

    # NRMSE, PSNR and SSIM
    rec_nrmse = nrmse_evaluate(img_norm, rec_norm)
    rec_psnr = psnr_evaluate(img_norm, rec_norm)
    alias_psnr = psnr_evaluate(img_norm, aliasing_norm)

    img_norm = img_norm.cpu().numpy()
    rec_norm = rec_norm.cpu().numpy()
    rec_ssim = ssim_evaluate(img_norm, rec_norm)

    psnr_m = torch.mean(rec_psnr)
    ssim_m = np.mean(rec_ssim)

    print(f"Mean NRMSE of reconstructed images: {rec_nrmse:.6f}")
    print(f"Mean PSNR of reconstructed images: {psnr_m:.6f}")
    print(f"Mean SSIM of reconstructed images: {ssim_m:.6f}")

    output_dict['echo'] = echo_norm.cpu().numpy()
    output_dict['img'] = img_norm
    output_dict['alias'] = aliasing_norm.cpu().numpy()
    output_dict['rec'] = rec_norm
    output_dict['rec_psnr'] = rec_psnr.cpu().numpy()
    output_dict['alias_psnr'] = alias_psnr.cpu().numpy()

    return output_dict


def figure_generate(output_dict, index, down_rate):
    save_dir = os.path.join('./images', args.regularization, f'{int(down_rate * 100)}pct')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if len(index) == 1:
        save_path = os.path.join(save_dir, f'figure_{1}.png')
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(output_dict['echo'][index[0]], cmap="gray", origin="lower")
        plt.title('Original Echo')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(output_dict['img'][index[0]], cmap="gray", origin="lower")
        plt.title(f'Original SLC Image')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(output_dict['alias'][index[0]], cmap="gray", origin="lower")
        alias_psnr = round(output_dict['alias_psnr'][index[0]], 2)
        plt.title(f'Aliasing Image, PSNR={alias_psnr:.2f} dB')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(output_dict['rec'][index[0]], cmap="gray", origin="lower")
        rec_psnr = round(output_dict['rec_psnr'][index[0]], 2)
        plt.title(f'Reconstructed Image, PSNR={rec_psnr:.2f} dB')
        plt.axis('off')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        for num in index:
            save_path = os.path.join(save_dir, f'figure_{num + 1}.png')
            plt.figure(figsize=(10, 10))

            plt.subplot(2, 2, 1)
            plt.imshow(output_dict['echo'][num], cmap="gray", origin="lower")
            plt.title('Original Echo')
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(output_dict['img'][num], cmap="gray", origin="lower")
            plt.title(f'Original SLC Image')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(output_dict['alias'][num], cmap="gray", origin="lower")
            alias_psnr = round(output_dict['alias_psnr'][num], 2)
            plt.title(f'Aliasing Image, PSNR={alias_psnr:.2f} dB')
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(output_dict['rec'][num], cmap="gray", origin="lower")
            rec_psnr = round(output_dict['rec_psnr'][num], 2)
            plt.title(f'Reconstructed Image, PSNR={rec_psnr:.2f} dB')
            plt.axis('off')

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def main(args):
    weight_path = args.weight
    device_index = args.device
    network = args.network
    batch_size = args.batch_size
    down_rate = args.down_rate

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = 0

    down_matrix, up_matrix = random_sampling_create(down_rate, device_index)

    test_image = torch.tensor(np.load(args.test_image), dtype=torch.complex64).to(device)
    test_echo = torch.tensor(np.load(args.test_echo), dtype=torch.complex64).to(device)

    test_dataset = TensorDataset(test_image, test_echo)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('DataLoader Finished!')

    if network == 'arsar':
        model = ARSARNet(
            device, 
            processor, 
            down_matrix, 
            up_matrix, 
            args.layer_num, 
            args.regularization,
            ).to(device)
    elif network == 'pnp':
        model = NonInversionADMMPnPNet(
            device_index, 
            processor, 
            up_matrix, 
            args.layer_num, 
            args.internal_iteration,
            args.regularization,
            ).to(device)
    elif network == 'sr':
        model = SRNet(
            processor,       
            device, 
            down_matrix, 
            up_matrix, 
            args.layer_num,  
            mode='plus',
            ).to(device)
    else:
        raise ValueError(f'Unknown network name found: {network}!')
    print('Model Initialized!')

    with warnings.catch_warnings():
        # avoid FutureWarning for 'weights_only=False'
        warnings.simplefilter("ignore", category=FutureWarning)
        model.load_state_dict(torch.load(weight_path, map_location=device)['model_state_dict'])
    model.eval()
    print('Weight File Loaded!')

    index = [i for i in range(8)]
    rec = test_model(model, test_loader, device, 
                     batch_size, test_image)
    output_dict = pre_process(rec, test_echo, test_image, up_matrix)
    figure_generate(output_dict, index, down_rate)


if __name__ == '__main__':
    args = get_args()
    main(args)
