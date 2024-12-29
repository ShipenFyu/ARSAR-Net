import h5py
import numpy as np

mat_file_path = ['./data/Training_DataX.mat', './data/Training_DataY.mat']
npy_file_path = ['./data/image_train.npy', './data/echo_train.npy', 
                 './data/image_test.npy', './data/echo_test.npy']


def mat2np():
    with h5py.File(mat_file_path[0], 'r') as f:
        image_labels = f['Training_DataX']
        print(f'Image Shape: {image_labels.shape}')
        image_labels = np.transpose(image_labels, (2, 1, 0))[:512, :, :]
        real_part = image_labels['real']
        imag_part = image_labels['imag']

        image_np = np.array(real_part, dtype=np.float32) + 1j * np.array(imag_part, dtype=np.float32)
        image_labels_array = np.array(image_np, dtype=np.complex64)
        
        np.save(npy_file_path[0], image_labels_array)

    with h5py.File(mat_file_path[1], 'r') as f:
        echo_labels = f['Training_DataY']
        print(f'Echo Shape: {echo_labels.shape}')
        echo_labels = np.transpose(echo_labels, (2, 1, 0))[:512, :, :]
        real_part = echo_labels['real']
        imag_part = echo_labels['imag']

        echo_np = np.array(real_part, dtype=np.float32) + 1j * np.array(imag_part, dtype=np.float32)
        echo_labels_array = np.array(echo_np, dtype=np.complex64)
        
        np.save(npy_file_path[1], echo_labels_array)
    
    print('Finished!')


def test_clip():
    image_data = np.load(npy_file_path[0])
    test_image_clip = image_data[:8]

    np.save(npy_file_path[2], test_image_clip)

    echo_data = np.load(npy_file_path[1])
    test_echo_clip = echo_data[:8]

    np.save(npy_file_path[3], test_echo_clip)

    print("Finished!")


if __name__ == '__main__':
    mat2np()
    test_clip()