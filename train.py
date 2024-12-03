import os
import argparse
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import time
from tqdm import tqdm

from utils.observation_matrix import get_ob_matrix
from utils.config import device_index
from models.lr_net import ADMMIRNet
from models.pnp_net import ADMMPnPNet
from logs.log_utils import configure_logging, log_training_info


parser = argparse.ArgumentParser(description='Implicit Regularization Training')
parser.add_argument('--trn_dataset', default='data', help='Training dataset directory')
parser.add_argument('--network', default='ir', help='Backbone network pnp or ir')
parser.add_argument('--epochs', default=40, type=int, help='Epochs')
parser.add_argument('--nsave', default=2, help='Save model after every nSave epoch')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate')
parser.add_argument('--layer_num', default=9, type=int, help='Net block num in iteration')
parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
parser.add_argument('--checkpoint', default=None, help='Continue model training')
parser.add_argument('--regularization', default='l1', help='The regularization type to PnP network')
parser.add_argument('--device', default=device_index, help='The regularization type to PnP network')

args = parser.parse_args()

network = args.network
epochs = args.epochs
batch_size = args.batch_size

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 0  # workers error

train_file_path = [os.path.join(args.trn_dataset, 'image_train.npy'), 
                   os.path.join(args.trn_dataset, 'echo_train.npy')]

image_labels_tensor = torch.tensor(np.load(train_file_path[0]), dtype=torch.complex64).to(device)
echo_labels_tensor = torch.tensor(np.load(train_file_path[1]), dtype=torch.complex64).to(device)

train_dataset = TensorDataset(image_labels_tensor, echo_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('DataLoader Finished!')

weights_dir = os.path.join('./weights', datetime.now().strftime("%Y_%m_%d"))
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

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


class NRMSE(nn.Module):
    def __init__(self):
        super(NRMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, target):
        # output and target are both complex matrix
        output = abs(output)
        target = abs(target)

        scale = torch.linalg.matrix_norm(target)
        nrmse = self.mse_loss(output, target) / torch.sqrt(torch.mean(scale))

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
    save_model = os.path.join(weights_dir, f'weights_model_{network}_epochs_{epoch}_' + 
                             f'{datetime.now().strftime("%H_%M_%S")}.pt')
    torch.save(checkpoint, save_model)


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
        for clip in tqdm(train_loader, desc=f'Training Epoch: [{epoch + 1}/{epochs}]'):
            image, echo = clip[0].to(device), clip[1].to(device)
            output = model(echo)

            optimizer.zero_grad()
            loss = criterion(output, image)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())

        avgTrnLoss = np.mean(epoch_loss)
        print(f'Loss: {avgTrnLoss:.6f}')
        log_training_info(logger, f'Epoch [{epoch + 1}/{args.epochs}], Loss: {avgTrnLoss:.6f}')

        if (epoch + 1) % args.nsave == 0:
            save_checkpoint(model, optimizer, epoch + 1, avgTrnLoss)

    end_time = time.time()
    print('Training completed in minutes', (end_time - start_time) / 60)
    print('Training completed at', datetime.now().strftime("%Y %m %d-%H:%M:%S"))
    log_training_info(logger, "Training completed")


if __name__ == '__main__':
    checkpoint = args.checkpoint
    train(checkpoint)
