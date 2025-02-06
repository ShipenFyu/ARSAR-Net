import os
import argparse
from typing import Iterable

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
from models.arsar_net import ARSARNet
from models.pnp_net import NonInversionADMMPnPNet
from logs.log_utils import configure_logging, log_training_info


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
    

class ComplexLoss(nn.Module):
    def __init__(self, batch_size):
        super(ComplexLoss, self).__init__()
        self.batch_size = batch_size

    def forward(self, output, target):
        residual = output - target
        loss = 0
        for index in range(self.batch_size):
            residual_norm = torch.linalg.matrix_norm(residual[index], ord=2)
            target_norm = torch.linalg.matrix_norm(target[index], ord=2)
            loss += residual_norm / target_norm 
        loss_average = loss / self.batch_size

        return loss_average


def get_args():
    parser = argparse.ArgumentParser(description='ARSAR-Net Training')
    parser.add_argument('--trn_dataset', default='./data/concat', help='Training dataset directory')
    parser.add_argument('--val_size', default=400, type=int, help='Validation dataset size')
    parser.add_argument('--device', default='cuda:0', help='The device index used in training')
    parser.add_argument('--network', default='arsar', help='Backbone network (pnp or arsar)')
    parser.add_argument('--criterion', default='norm', help='Criterion type (mse or norm)')
    parser.add_argument('--regularization', default='haar', help='The regularization type in ARSAR-Net')
    parser.add_argument('--epochs', default=100, type=int, help='Epochs')
    parser.add_argument('--nsave', default=1, type=int, help='Save model after every nSave epoch')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate')
    parser.add_argument('--layer_num', default=9, type=int, help='Net block num in iteration')
    parser.add_argument('--internal_iteration', default=6, type=int, help='ADMM-Net z block iteration num')
    parser.add_argument('--checkpoint', default=None, help='Continue model training')
    parser.add_argument('--down_rate', default=0.5, type=float, help='Azimuth down-sampling rate')
    args = parser.parse_args()

    return args


def save_checkpoint(model_cur, optimizer_cur, epoch, loss, weights_dir, down_rate):
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


def train_epochs(model: nn.Module, criterion: nn.Module,
          train_loader: Iterable, val_loader: Iterable, 
          optimizer: optim.Optimizer, device: torch.device, 
          epochs: int, batch_size: int, down_rate: float, 
          label: str, regular: str, weights_dir: str,
          logger, checkpoint=None
          ):
    log_training_info(logger, 'Training started')
    log_training_info(logger, f"Parameters: Epochs: {epochs}, Batch Size: {batch_size}, "
                              f"Learning Rate: {args.lr}, Regularization: {regular}")
    log_training_info(logger, f"Scene: {label}, criterion: {args.criterion}")
    log_training_info(logger, f"Layer_num: {args.layer_num}, "
                              f"Internal_iteration(if recurrent): {args.internal_iteration}")
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
                save_checkpoint(model, optimizer, epoch + 1, avgTrnLoss, weights_dir, down_rate)

    end_time = time.time()
    print('Training completed in minutes', (end_time - start_time) / 60)
    print('Training completed at', datetime.now().strftime("%Y %m %d-%H:%M:%S"))
    log_training_info(logger, f"Training completed in minutes {(end_time - start_time) / 60}")
    log_training_info(logger, f"Training completed at {datetime.now().strftime('%Y %m %d-%H:%M:%S')}")


def main(args):
    device_index = args.device
    network = args.network
    criterion_type = args.criterion
    regular = args.regularization
    epochs = args.epochs
    batch_size = args.batch_size
    down_rate = args.down_rate
    checkpoint = args.checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = 0

    _, up_matrix = random_sampling_create(down_rate, device_index)

    train_file_path = [os.path.join(args.trn_dataset, 'image_train.npy'), 
                    os.path.join(args.trn_dataset, f'echo_{int(down_rate * 100)}_train.npy')]

    train_image = torch.tensor(np.load(train_file_path[0]), dtype=torch.complex64).to(device)
    train_echo = torch.tensor(np.load(train_file_path[1]), dtype=torch.complex64).to(device)

    val_size = args.val_size
    random_indices = torch.randperm(len(train_image))

    val_image = train_image[random_indices[:val_size]]
    val_echo = train_echo[random_indices[:val_size]]

    train_dataset = TensorDataset(train_image, train_echo)
    val_dataset = TensorDataset(val_image, val_echo)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print('DataLoader Finished!')

    label = args.trn_dataset.split('/')[-1]
    weights_dir = os.path.join('../Dataset/FuShiping/weights', label, datetime.now().strftime("%Y_%m_%d"))
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    if network == 'arsar':
        model = ARSARNet( 
            device_index, 
            processor, 
            up_matrix, 
            args.layer_num, 
            args.internal_iteration,
            regular,
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
        raise ValueError(f'Unknown network name found: {network}!')
    print('Model Initialized!')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if criterion_type == 'mse':
        criterion = NRMSE()
    elif criterion_type == 'norm':
        criterion = ComplexLoss(batch_size)
    else:
        raise ValueError(f'Unknown criterion type found: {criterion_type}!')
    logger = configure_logging(network, epochs)

    train_epochs(model, criterion, train_loader, val_loader, 
                 optimizer, device, epochs, batch_size, down_rate, 
                 label, regular, weights_dir, logger, checkpoint)


if __name__ == '__main__':
    args = get_args()
    main(args)
