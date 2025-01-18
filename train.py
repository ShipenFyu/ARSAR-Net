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

from utils.observation_matrix import random_sampling_create
from utils.config import processor
from models.ir_net import ADMMIRNet
from models.pnp_net import NonInversionADMMPnPNet
from logs.log_utils import configure_logging, log_training_info


parser = argparse.ArgumentParser(description='ARSAR-Net Training')
parser.add_argument('--trn_dataset', default='./data/train', help='Training dataset directory')
parser.add_argument('--val_dataset', default='./data/val', help='Validation dataset directory')
parser.add_argument('--device', default='cuda:1', help='The regularization type to PnP network')
parser.add_argument('--network', default='pnp', help='Backbone network pnp or ir')
parser.add_argument('--regularization', default='unet', help='The regularization type to PnP network')
parser.add_argument('--epochs', default=80, type=int, help='Epochs')
parser.add_argument('--nsave', default=1, type=int, help='Save model after every nSave epoch')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate')
parser.add_argument('--layer_num', default=8, type=int, help='Net block num in iteration')
parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
parser.add_argument('--checkpoint', default=None, help='Continue model training')
parser.add_argument('--down_rate', default=0.50, type=float, help='Azimuth down-sampling rate')

args = parser.parse_args()

device_index = args.device
network = args.network
regular = args.regularization
epochs = args.epochs
batch_size = args.batch_size
down_rate = args.down_rate

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
if platform.system() == 'Windows':
    num_workers = 0
else:
    num_workers = 0  # workers error

_, up_matrix = random_sampling_create(down_rate, device_index)

train_file_path = [os.path.join(args.trn_dataset, 'image_train.npy'), 
                   os.path.join(args.trn_dataset, f'echo_{int(down_rate * 100)}_train.npy')]
val_file_path = [os.path.join(args.val_dataset, 'image_val.npy'), 
                 os.path.join(args.val_dataset, f'echo_{int(down_rate * 100)}_val.npy')]

train_image = torch.tensor(np.load(train_file_path[0]), dtype=torch.complex64).to(device)
train_echo = torch.tensor(np.load(train_file_path[1]), dtype=torch.complex64).to(device)

val_image = torch.tensor(np.load(train_file_path[0]), dtype=torch.complex64).to(device)
val_echo = torch.tensor(np.load(train_file_path[1]), dtype=torch.complex64).to(device)

train_dataset = TensorDataset(train_image, train_echo)
val_dataset = TensorDataset(val_image, val_echo)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('DataLoader Finished!')

weights_dir = os.path.join('./weights', datetime.now().strftime("%Y_%m_%d"))
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

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
        up_matrix, 
        args.layer_num, 
        args.internal_iteration,
        regular,
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
    save_model = os.path.join(weights_dir, f'downsample_{down_rate}_epochs_{epoch}_' + 
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
                              f" Learning Rate: {args.lr}, Regularization: {regular}")
    log_training_info(logger, f"Layer_num: {args.layer_num}, Internal_iteration: {args.internal_iteration}")
    log_training_info(logger, f"Downsampling Rate: {int(down_rate * 100)}percent")
    log_training_info(logger, f"Training started at {datetime.now().strftime('%Y %m %d-%H:%M:%S')}")
    start_time = time.time()
    print('Training started at', datetime.now().strftime("%Y %m %d-%H:%M:%S"))

    start_epoch = 0
    if checkpoint is not None:
        if os.path.isfile(checkpoint):
            start_epoch = load_checkpoint(model, optimizer, checkpoint)
    
    best_val_loss = float('inf')
    counter = 0
    for epoch in range(start_epoch, epochs):
        model.train()
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
        print(f'Training Loss: {avgTrnLoss:.6f}')
        log_training_info(logger, f'Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avgTrnLoss:.6f}')

        model.eval()
        val_loss = []
        with torch.no_grad():
            for clip in tqdm(val_loader, desc=f'Validation Epoch: [{epoch + 1}/{epochs}]'):
                image, echo = clip[0].to(device), clip[1].to(device)
                output = model(echo)
                loss = criterion(output, image)
                val_loss.append(loss.item())

        avgValLoss = np.mean(val_loss)
        print(f'Validation Loss: {avgValLoss:.6f}')
        log_training_info(logger, f'Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {avgValLoss:.6f}')

        if avgValLoss < best_val_loss:
            counter += 1
            best_val_loss = avgValLoss
            if counter % args.nsave == 0:
                print(f'Saving model with improved validation loss: {best_val_loss:.6f}')
                log_training_info(logger, f'Saving model with improved validation loss: {best_val_loss:.6f}')
                save_checkpoint(model, optimizer, epoch + 1, avgTrnLoss)

    end_time = time.time()
    print('Training completed in minutes', (end_time - start_time) / 60)
    print('Training completed at', datetime.now().strftime("%Y %m %d-%H:%M:%S"))
    log_training_info(logger, f"Training completed in minutes {(end_time - start_time) / 60}")
    log_training_info(logger, f"Training completed at {datetime.now().strftime('%Y %m %d-%H:%M:%S')}")
    

if __name__ == '__main__':
    checkpoint = args.checkpoint
    train(checkpoint)
