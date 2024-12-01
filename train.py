import os
import argparse
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
from datetime import datetime
import time
from tqdm import tqdm

from utils.observation_matrix import get_ob_matrix
from models.admm_net import ADMMBasicNet
from logs.log_utils import configure_logging, log_training_info


parser = argparse.ArgumentParser(description='Implicit Regularization Training')
parser.add_argument('--trn_dataset', default='data', help='Training dataset directory')
parser.add_argument('--network', default='ir', help='Backbone network pnp or ir')
parser.add_argument('--epochs', default=80, type=int, help='Epochs')
parser.add_argument('--nsave', default=5, help='Save model after every nSave epoch')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
parser.add_argument('--layer_num', default=7, type=int, help='Net block num in iteration')
parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
parser.add_argument('--gpu_list', type=str, default='0', help='GPU index')

args = parser.parse_args()

network = args.network
epochs = args.epochs
batch_size = args.batch_size

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 0

training_image = 'Training_DataX.mat'
with h5py.File('./%s/%s' % (args.trn_dataset, training_image), 'r') as f:
    image_labels = f['Training_DataX']
    image_labels = np.transpose(image_labels, (2, 1, 0))[:512, :, :]  # we need only 512 samples here
    real_part = image_labels['real']
    imag_part = image_labels['imag']

    image_labels_tensor = torch.complex(torch.tensor(real_part, dtype=torch.float32), 
                                            torch.tensor(imag_part, dtype=torch.float32)).to(device)

training_echo = 'Training_DataY.mat'
with h5py.File('./%s/%s' % (args.trn_dataset, training_echo), 'r') as f:
    echo_labels = f['Training_DataY']
    echo_labels = np.transpose(echo_labels, (2, 1, 0))[:512, :, :]
    real_part = echo_labels['real']
    imag_part = echo_labels['imag']

    echo_labels_tensor = torch.complex(torch.tensor(real_part, dtype=torch.float32), 
                                            torch.tensor(imag_part, dtype=torch.float32)).to(device)

train_dataset = TensorDataset(image_labels_tensor, echo_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

weights_dir = os.path.join('./weights', datetime.now().strftime("%Y_%m_%d"))
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

right_matrix, left_matrix, operator = get_ob_matrix(batch_size)

if network == 'ir':
    model = ADMMBasicNet(left_matrix, right_matrix, operator, args.layer_num, args.internal_iteration).to(device)
else:
    raise ValueError(f'unknown network name {network} found!')


class NRMSE(nn.Module):
    def __init__(self):
        super(NRMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # output and target are both complex matrix
        output = abs(output)
        target = abs(target)

        scale = torch.linalg.matrix_norm(target)
        nrmse = self.mse_loss(output, target) / torch.mean(scale)

        return nrmse


optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = NRMSE()
logger = configure_logging(network, epochs)


def save_checkpoint(model_cur, optimizer_cur, epoch, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_cur.state_dict(),
        'optimizer_state_dict': optimizer_cur.state_dict(),
        'loss': loss,
    }
    savemodel = os.path.join(weights_dir, f'weights_model_{network}_epochs_{epochs}' + 
                             f'{datetime.now().strftime("%H_%M_%S")}.pt')
    torch.save(checkpoint, savemodel)


def load_checkpoint(model_cur, optimizer_cur, filename):
    checkpoint = torch.load(filename)
    model_cur.load_state_dict(checkpoint['model_state_dict'])
    optimizer_cur.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filename}. Resuming from epoch {epoch}")

    return epoch


def train(checkpoint=None):
    log_training_info(logger, 'Training started')
    log_training_info(logger, 'Platform: {}'.format(platform.system()))
    log_training_info(logger, f"Parameters: Epochs: {epochs}, Batch Size: {batch_size},"
                              f" Learning Rate: {args.lr}")
    start_time = time.time()
    print('Training started at', datetime.now().strftime("%Y %m %d-%H:%M:%S"))

    start_epoch = 0
    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            start_epoch = load_checkpoint(model, optimizer, checkpoint)
    
    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = []
        for clip in tqdm(train_loader):
            image, echo = clip[0].to(device), clip[1].to(device)
            output = model(echo)

            optimizer.zero_grad()
            loss = criterion(output, image)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avgTrnLoss = torch.mean(epoch_loss)
        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avgTrnLoss:.6f}')
        log_training_info(logger, f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avgTrnLoss:.6f}')

        if (epoch + 1) % args.nsave == 0:
            save_checkpoint(model, optimizer, epoch + 1, avgTrnLoss)

    end_time = time.time()
    print('Training completed in minutes', (end_time - start_time) / 60)
    print('Training completed at', datetime.now().strftime("%Y %m %d-%H:%M:%S"))
    log_training_info(logger, "Training completed")


if __name__ == '__main__':
    train()