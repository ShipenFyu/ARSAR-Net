import os
import argparse
import platform
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils.observation_matrix import downsampling_matrix_create
from utils.evaluate import normalize, psnr_evaluate, ssim_evaluate, aliasing_construct
from utils.config import processor
from models.ir_net import ADMMIRNet
from models.pnp_net import NonInversionADMMPnPNet


parser = argparse.ArgumentParser(description='Implicit Regularization Testing')
parser.add_argument('--tst_dataset', default='data', help='Testing dataset directory')
parser.add_argument('--device', default='cuda:0', help='The regularization type to PnP network')
parser.add_argument('--network', default='pnp', help='Backbone network pnp or ir')
parser.add_argument('--regularization', default='unet', help='The regularization type to PnP network')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for testing')
parser.add_argument('--layer_num', default=9, type=int, help='Net block num in iteration')
parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
parser.add_argument('--down_sampling_rate', default=0.5, type=float, help='Azimuth down-sampling rate')

args = parser.parse_args()

device_index = args.device
network = args.network
batch_size = args.batch_size
down_rate = args.down_sampling_rate

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 0  # workers error

down_matrix, down_matrix_t = downsampling_matrix_create(down_rate, device_index)

test_file_path = [os.path.join(args.tst_dataset, 'image_test.npy'), 
                   os.path.join(args.tst_dataset, 'echo_test.npy')]

image_labels_array = np.load(test_file_path[0])
echo_labels_array = np.load(test_file_path[1])

image_labels_tensor = torch.tensor(image_labels_array, dtype=torch.complex64).to(device)
echo_labels_tensor = torch.tensor(echo_labels_array, dtype=torch.complex64).to(device)
echo_labels_tensor = torch.einsum('ij,bjk->bik', down_matrix, echo_labels_tensor)  # echo downsampling

test_dataset = TensorDataset(image_labels_tensor, echo_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('DataLoader Finished!')

if network == 'ir':
    model = ADMMIRNet(
        processor, 
        args.layer_num, 
        args.internal_iteration,
        ).to(device)
elif network == 'pnp':
    model = NonInversionADMMPnPNet(
        device_index, 
        processor, 
        down_matrix_t, 
        args.layer_num, 
        args.internal_iteration,
        args.regularization,
        ).to(device)
else:
    raise ValueError(f'unknown network name {network} found!')
print('Model Initialized!')

split_path = '/home/FuShiping/ADMM-IR/weights/2024_12_26/weights_model_pnp_epochs_26_22_58_30.pt'.split('/')[-3:]
weight_path = os.path.join(split_path[0], split_path[1], split_path[2])

with warnings.catch_warnings():
    # avoid FutureWarning for 'weights_only=False'
    warnings.simplefilter("ignore", category=FutureWarning)
    model.load_state_dict(torch.load(weight_path)['model_state_dict'])
model.eval()
print('Weight File Loaded!')


def test():
    rec = np.zeros(image_labels_array.shape ,dtype=np.complex64)

    print('SAR Reconstruction started at', datetime.now().strftime("%H:%M:%S"))
    with torch.no_grad():
        for i, clip in enumerate(tqdm(test_loader, desc=f'Reconstruction')):
                _, echo = clip[0].to(device), clip[1].to(device)
                output = model(echo)
                rec[i * batch_size: (i + 1) * batch_size, :, :] = output.cpu().numpy()
    print('Reconstruction completed at', datetime.now().strftime("%H:%M:%S"))

    return rec


def pre_process(rec, echo, img):
    output_dict = {}

    echo = np.einsum('ij,bjk->bik', down_matrix.cpu().numpy(), echo)
    aliasing = aliasing_construct(echo, down_matrix_t.cpu().numpy(), processor)

    rec = np.abs(rec)
    echo = np.abs(echo)
    aliasing = np.abs(aliasing)
    img = np.abs(img)

    # norm
    rec_norm = normalize(rec)
    echo_norm = normalize(echo)
    aliasing_norm = normalize(aliasing)
    img_norm = normalize(img)

    # PSNR and SSIM
    rec_psnr = psnr_evaluate(img_norm, rec_norm)
    rec_ssim = ssim_evaluate(img_norm, rec_norm)

    alias_psnr = psnr_evaluate(img_norm, aliasing_norm)

    psnr_mean = np.mean(rec_psnr)
    ssim_mean = np.mean(rec_ssim)

    print(f"Mean PSNR of reconstructed images: {psnr_mean:.6f}")
    print(f"Mean SSIM of reconstructed images: {ssim_mean:.6f}")

    output_dict['echo'] = echo_norm
    output_dict['img'] = img_norm
    output_dict['alias'] = aliasing_norm
    output_dict['rec'] = rec_norm
    output_dict['rec_psnr'] = rec_psnr
    output_dict['alias_psnr'] = alias_psnr

    return output_dict


def figure_generate(output_dict, index):
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
        plt.title(f'Aliasing Image, PSNR={alias_psnr} dB')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(output_dict['rec'][index[0]], cmap="gray", origin="lower")
        rec_psnr = round(output_dict['rec_psnr'][index[0]], 2)
        plt.title(f'Reconstructed Image, PSNR={rec_psnr} dB')
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
            plt.title(f'Aliasing Image, PSNR={alias_psnr} dB')
            plt.axis('off')

            plt.subplot(2, 2, 4)
            plt.imshow(output_dict['rec'][num], cmap="gray", origin="lower")
            rec_psnr = round(output_dict['rec_psnr'][num], 2)
            plt.title(f'Reconstructed Image, PSNR={rec_psnr} dB')
            plt.axis('off')

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    index = [i for i in range(8)]
    rec = test()
    output_dict = pre_process(rec, echo_labels_array, image_labels_array)
    figure_generate(output_dict, index)
