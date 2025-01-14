import os
import sys
import numpy as np
import mat73
from PIL import Image
from tqdm import tqdm
from osgeo import gdal
from numpy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt

from utils.config import processor
from utils.observation_matrix import downsampling_matrix_create
from utils.evaluate import aliasing_construct, normalize, psnr_evaluate


def clip_slc_dataset(slc_dir, saved_dir):
    dataset = gdal.Open(slc_dir)
    os.makedirs(saved_dir, exist_ok=True)

    band_real = dataset.GetRasterBand(1)
    band_imag = dataset.GetRasterBand(2)

    data_real = band_real.ReadAsArray()
    data_imag = band_imag.ReadAsArray()

    height, width = data_real.shape

    complex_data = data_real + 1j * data_imag
    np.save(os.path.join(saved_dir, 'complex_data.npy'), complex_data)
    print('Total data saved!')

    patch_size = 512

    for idx_y, y in enumerate(range(0, height, patch_size)):
        for idx_x, x in tqdm(enumerate(range(0, width, patch_size))):
            if y + patch_size <= height and x + patch_size <= width:
                patch = np.abs(complex_data[y:y + patch_size, x:x + patch_size])

                patch_image = Image.fromarray(patch)
                patch_image = patch_image.convert('L')

                patch_filename = f'{idx_y}_{idx_x}.jpg'
                patch_image.save(os.path.join(saved_dir, patch_filename))
    print("Finished!")


def select_ship_dataset():
    mat_file_path = './data/ship/Training_DataX.mat'
    saved_dir = './data/ship/images'
    os.makedirs(saved_dir, exist_ok=True)

    image_data = mat73.loadmat(mat_file_path)['labels']
    image_labels_array = np.array(image_data, dtype=np.complex64)
    np.save(os.path.join('./data/ship', 'complex_data.npy'), image_labels_array)
        
    for i in tqdm(range(len(image_labels_array))):
        patch = np.abs(image_labels_array[i])

        patch_image = Image.fromarray(patch)
        patch_image = patch_image.convert('L')

        patch_filename = f'{i}.jpg'
        patch_image.save(os.path.join(saved_dir, patch_filename))
    
    print('Finished!')


def organize_dataset(cls, input_dir, data):
    if cls == 'ship':
        image_folder = os.path.join(input_dir, 'images')
        output_npy_file = os.path.join(input_dir, 'selected_data.npy')

        jpg_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        jpg_files.sort(key=lambda x: int(x.split('.')[0]))

        slices = []
        for jpg_file in jpg_files:
            index = int(jpg_file.split('.')[0])
            
            slice_data = data[index]
            slices.append(slice_data)

        combined_data = np.stack(slices, axis=0)

        np.save(output_npy_file, combined_data)
    else:
        output_npy_file = os.path.join(input_dir, 'selected_data.npy')

        jpg_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]

        slices = []
        for jpg_file in jpg_files:
            location = jpg_file.split('.')[0]
            y, x = int(location.split('_')[0]), int(location.split('_')[1])
            
            slice_data = data[y*512: (y+1)*512, x*512: (x+1)*512]
            slices.append(slice_data)

        combined_data = np.stack(slices, axis=0)

        np.save(output_npy_file, combined_data)


def visualize(data):
    saved_dir = './data/images'
    os.makedirs(saved_dir, exist_ok=True)
    for i in tqdm(range(len(data))):
        patch = np.abs(data[i])

        patch_image = Image.fromarray(patch)
        patch_image = patch_image.convert('L')

        patch_filename = f'{i}.jpg'
        patch_image.save(os.path.join(saved_dir, patch_filename))
    
    print('Finished!')


def stack_dataset(pad):
    dataset_dir = './data/dataset'
    output_npy_file = './data/dataset/dataset.npy'

    slices = []
    for file in os.listdir(dataset_dir):
        data = np.load(os.path.join(dataset_dir, file))
        for i in range(len(data)):
            slices.append(data[i])

    if pad:
        pad_slices = []
        random_indices = np.random.choice(960, 40, replace=False)
        for num in random_indices:
            pad_slices.append(slices[num])
        for j in range(len(pad_slices)):
            slices.append(pad_slices[j])

    total_data = np.stack(slices, axis=0)
    np.random.shuffle(total_data)
    
    print(total_data.shape)
    np.save(output_npy_file, total_data)


def split_dataset():
    dataset_file = './data/dataset/dataset.npy'
    train_file = './data/dataset/train_dataset.npy'
    val_file = './data/dataset/val_dataset.npy'
    test_file = './data/dataset/test_dataset.npy'

    data = np.load(dataset_file)
    np.random.shuffle(data)
    
    train_data = data[:1200]
    val_data = data[1200:1500]
    test_data = data[1500:1800]

    np.save(train_file, train_data)
    np.save(val_file, val_data)
    np.save(test_file, test_data)

    print('Finished!')


class EchoProcessor:
    def __init__(self, down_rate):
        self.down_rate = down_rate
        self.down_matrix, self.up_matrix = downsampling_matrix_create(down_rate, 'cuda:0')

    def imaging_inverse(self, image, label):
        saved_file = os.path.join('./data', label, f'echo_{self.down_rate}_{label}.npy')

        image_fr = fftshift(fft(image, axis=1, norm='ortho'), axes=1)
        image_fr = image_fr * np.conj(processor['ac'].numpy())
        image_fra = fftshift(fft(image_fr, axis=2, norm='ortho'), axes=2)
        image_fra = image_fra * np.conj(processor['rc'].numpy())
        image_fr = ifft(ifftshift(image_fra, axes=2), axis=2, norm='ortho')
        image_fr = image_fr * np.conj(processor['sc'].numpy())
        image_fr = ifftshift(image_fr, axes=1)
        echo = np.einsum('ij,bjk->bik', self.down_matrix.cpu().numpy, image_fr)
        np.save(saved_file, echo)
        print('Finished!')

        return echo
    
    def sample_test(self, image, label):
        index = [i for i in range(5)]

        echo = self.imaging_inverse(image, label)
        aliasing = aliasing_construct(echo, self.up_matrix, processor)

        img_norm = normalize(np.abs(image))
        echo_norm = normalize(np.abs(echo))
        aliasing_norm = normalize(np.abs(aliasing))

        aliasing_psnr = psnr_evaluate(img_norm, aliasing_norm)

        for num in index:
            plt.figure(figsize=(10, 10))

            plt.subplot(2, 2, 1)
            plt.imshow(echo_norm[num], cmap="gray", origin="lower")
            plt.title('Original Echo')
            plt.axis('off')

            plt.subplot(2, 2, 2)
            plt.imshow(img_norm[num], cmap="gray", origin="lower")
            plt.title(f'Original SLC Image')
            plt.axis('off')

            plt.subplot(2, 2, 3)
            plt.imshow(aliasing_norm[num], cmap="gray", origin="lower")
            alias_psnr = round(aliasing_psnr[num], 2)
            plt.title(f'Aliasing Image, PSNR={alias_psnr} dB')
            plt.axis('off')

            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
            plt.show()


if __name__ == '__main__':
    # slc_dir = './data/GF3_L1A_harbour/GF3_MYN_UFS_013470_E109.5_N18.4_20190302_L1A_DH_L10000000015/GF3_MYN_UFS_013470_E109.5_N18.4_20190302_L1A_DH_L10000000015.tiff'
    # saved_dir = './data/harbour/GF3_MYN_UFS_013470_E109.5_N18.4_20190302_L1A_DH_L10000000015'
    # clip_slc_dataset(slc_dir, saved_dir)

    # select_ship_dataset()

    # npy_file = './data/harbour/GF3_MYN_UFS_013470_E109.5_N18.4_20190302_L1A_DH_L10000000015/complex_data.npy'
    # image_dir = './data/harbour/GF3_MYN_UFS_013470_E109.5_N18.4_20190302_L1A_DH_L10000000015'
    # data = np.load(npy_file)
    # organize_dataset('harbour', image_dir, data)

    # load_file = './data/harbour/GF3_MYN_UFS_013470_E109.5_N18.4_20190302_L1A_DH_L10000000015/selected_data.npy'
    # load_file = './data/dataset/val_dataset.npy'
    # load_data = np.load(load_file)
    # visualize(load_data)

    # dataset_dir = './data/dataset'
    # for file in os.listdir(dataset_dir):
    #     data = np.load(os.path.join(dataset_dir, file))
    #     print(data.shape)

    # stack_dataset(False)
    # split_dataset()

    down_rate = 1.0
    label = 'train'
    file_path = os.path.join('./data', label, f'image_{label}.npy')
    image = np.load(file_path)
    
    echo_processor = EchoProcessor(down_rate)
    echo_processor.sample_test(image, label)
