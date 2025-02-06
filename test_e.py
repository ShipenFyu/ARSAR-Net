import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils.observation_matrix import random_sampling_create
from utils.evaluate import range_cut, psnr_evaluate, ssim_evaluate, aliasing_construct
from utils.config import processor
from models.solver import ista_l1_one_object, admm_tv_one_object


def get_args():
    parser = argparse.ArgumentParser(description='Explicit Regularization Testing')
    parser.add_argument('--tst_dataset', default='./data/concat', help='Testing dataset directory')
    parser.add_argument('--device', default='cuda:0', help='The device index used in testing')
    parser.add_argument('--regularization', default='l1', help='The regularization type like l1, tv')
    parser.add_argument('--down_rate', default=0.5, type=float, help='Azimuth down-sampling rate')
    args = parser.parse_args()

    return args


def test_algorithm(test_image, test_echo, regular, down_matrix, up_matrix, device_index):
    rec = np.zeros(test_image.shape ,dtype=np.complex64)
    image_example = rec[0]

    print('SAR Reconstruction started at', datetime.now().strftime("%H:%M:%S"))
    for i, echo in tqdm(enumerate(test_echo), desc=f'Reconstruction'):
        if regular == 'l1':
            rst = ista_l1_one_object(image_example, echo, processor, 
                                     down_matrix, up_matrix, device_index)
        elif regular == 'tv':
            rst = admm_tv_one_object(image_example, echo, processor, up_matrix, device_index)
        else:
            raise ValueError(f'Unknown regularization type found: {regular}!')
        rec[i] = rst.cpu().numpy()
    print('Reconstruction completed at', datetime.now().strftime("%H:%M:%S"))

    return rec


def pre_process(rec, echo, img, up_matrix):
    output_dict = {}
    aliasing = aliasing_construct(echo, up_matrix.cpu().numpy(), processor)

    rec = np.abs(rec)
    echo = np.abs(echo)
    aliasing = np.abs(aliasing)
    img = np.abs(img)

    # range cut
    rec_norm = range_cut(rec)
    echo_norm = range_cut(echo)
    aliasing_norm = range_cut(aliasing)
    img_norm = range_cut(img)

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


def main(args):
    device_index = args.device
    regular = args.regularization
    down_rate = args.down_rate

    down_matrix, up_matrix = random_sampling_create(down_rate, device_index)

    test_file_path = [os.path.join(args.tst_dataset, 'image_test.npy'), 
                      os.path.join(args.tst_dataset, f'echo_{int(down_rate * 100)}_test.npy')]

    test_image_array = np.load(test_file_path[0])
    test_echo_array = np.load(test_file_path[1])
    test_echo = torch.tensor(test_echo_array, dtype=torch.complex64)
    print('DataLoader Finished!')

    index = [i for i in range(8)]
    rec = test_algorithm(test_image_array, test_echo, regular, 
                         down_matrix, up_matrix, device_index)
    output_dict = pre_process(rec, test_echo_array, test_image_array, up_matrix)
    figure_generate(output_dict, index, down_rate)


if __name__ == '__main__':
    args = get_args()
    main(args)
