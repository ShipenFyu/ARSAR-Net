import os
import argparse
import platform

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils.observation_matrix import get_ob_matrix
from utils.evaluate import normalize, psnr_evaluate, ssim_evaluate
from utils.config import device_index
from models.lr_net import ADMMIRNet
from models.pnp_net import ADMMPnPNet


parser = argparse.ArgumentParser(description='Implicit Regularization Testing')
parser.add_argument('--tst_dataset', default='data', help='Testing dataset directory')
parser.add_argument('--network', default='ir', help='Backbone network pnp or ir')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for testing')
parser.add_argument('--layer_num', default=9, type=int, help='Net block num in iteration')
parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
parser.add_argument('--regularization', default='l1', help='The regularization type to PnP network')
parser.add_argument('--device', default=device_index, help='The regularization type to PnP network')

args = parser.parse_args()

network = args.network
batch_size = args.batch_size

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 0  # workers error

test_file_path = [os.path.join(args.tst_dataset, 'image_test.npy'), 
                  os.path.join(args.tst_dataset, 'echo_test.npy')]

image_labels_array = np.load(test_file_path[0])
echo_labels_array = np.load(test_file_path[1])

image_labels_tensor = torch.tensor(image_labels_array, dtype=torch.complex64).to(device)
echo_labels_tensor = torch.tensor(echo_labels_array, dtype=torch.complex64).to(device)

test_dataset = TensorDataset(image_labels_tensor, echo_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('DataLoader Finished!')

right_matrix, left_matrix, operator = get_ob_matrix(batch_size)

if network == 'ir':
    model = ADMMIRNet(
        left_matrix, 
        right_matrix, 
        operator, 
        args.layer_num, 
        args.internal_iteration,
        ).to(device)
elif network == 'pnp':
    model = ADMMPnPNet(
        left_matrix, 
        right_matrix, 
        operator, 
        args.layer_num, 
        args.internal_iteration,
        args.regularization,
        ).to(device)
else:
    raise ValueError(f'unknown network name {network} found!')
print('Model Initialized!')

weights_dir = os.path.join('./weights', '2024_12_02')
weight_path = os.path.join(weights_dir, 'weights_model_ir_epochs_18_00_04_33.pt')

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
    rec = np.abs(rec)
    echo = np.abs(echo)
    img = np.abs(img)

    # norm
    rec_norm = normalize(rec)
    echo_norm = normalize(echo)
    img_norm = normalize(img)

    # PSNR and SSIM
    rec_psnr = psnr_evaluate(img_norm, rec_norm)
    rec_ssim = ssim_evaluate(img_norm, rec_norm)

    psnr_mean = np.mean(rec_psnr)
    ssim_mean = np.mean(rec_ssim)

    print(f"Mean PSNR of reconstructed images: {psnr_mean:.6f}")
    print(f"Mean SSIM of reconstructed images: {ssim_mean:.6f}")
    return echo_norm, rec_norm, img_norm, rec_psnr


def figure_generate(echo_norm, rec_norm, img_norm, rec_psnr, index):
    if len(index) == 1:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(echo_norm[index[0]], cmap="gray", origin="lower")
        plt.title('Original Echo')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(img_norm[index[0]], cmap="gray", origin="lower")
        plt.title(f'Original SLC image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(rec_norm[index[0]], cmap="gray", origin="lower")
        plt.title(f'Reconstructed image, PSNR={round(rec_psnr[index[0]], 2)} dB')
        plt.axis('off')

        plt.subplots_adjust(left=0, right=1, top=.95, bottom=0, wspace=0.01)
        
        save_path = f'./images/figure_{num}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        for num in index:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.imshow(echo_norm[num], cmap="gray", origin="lower")
            plt.title('Original Echo')
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(img_norm[num], cmap="gray", origin="lower")
            plt.title(f'Original SLC image')
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(rec_norm[num], cmap="gray", origin="lower")
            plt.title(f'Reconstructed image, PSNR={round(rec_psnr[num], 2)} dB')
            plt.axis('off')

            plt.subplots_adjust(left=0, right=1, top=.95, bottom=0, wspace=0.01)
            
            save_path = f'./images/figure_{num}.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


if __name__ == '__main__':
    index = [i for i in range(8)]
    rec = test()
    echo_norm, rec_norm, img_norm, rec_psnr = pre_process(rec, echo_labels_array, 
                                                          image_labels_array)
    figure_generate(echo_norm, rec_norm, img_norm, rec_psnr, index)
