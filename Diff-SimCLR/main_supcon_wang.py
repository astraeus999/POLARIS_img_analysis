import os
import sys
import argparse
import time
import math
import csv
from torch.utils.data import Dataset
from PIL import Image


import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter, TwoCropTransformFeature
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from resnet import SupConResNet
from losses import SupConLoss
import numpy as np

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


#### this is an unsupervised model for downstream tasks (SupCon)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='100,200,300',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')### changed
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=256, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SimCLR',
                        choices=['SupCon', 'SimCLR'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    if opt.data_folder is None:
        opt.data_folder = './images/'
    opt.model_path = './Result/your_folder'
    opt.tb_path = './Result/your_folder'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 1e-4
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, feature_transform=None):
        self.root = root
        self.transform = transform
        self.feature_transform = feature_transform
        self.image_paths = []
        self.feature_paths = []

        csv_path = os.path.join(root, "v_feat_mapping.csv")
        feature_folder = os.path.join(root, "features")

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                image_filename, feature_filename = row
                image_path = os.path.join(root, image_filename)
                feature_path = os.path.join(feature_folder, feature_filename)
                self.image_paths.append(image_path)
                self.feature_paths.append(feature_path)

        print("Image and feature file pairs:")
        for img, feat in zip(self.image_paths, self.feature_paths):
            print(f"Image: {os.path.basename(img)} <--> Feature: {os.path.basename(feat)}")

    def __len__(self):
        assert len(self.image_paths) == len(self.feature_paths), "Mismatched image and feature counts"
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        feat_path = self.feature_paths[idx]

        image = Image.open(img_path).convert('L')  # Change to 'RGB' if needed
        feature = np.load(feat_path)
        feature = torch.tensor(feature, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        if self.feature_transform:
            feature = self.feature_transform(feature)

        return image, feature



def set_loader_as(opt):
    # construct data loader for unlabeled data
    mean = (0.5,)
    std = (0.5,)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.4, contrast=0.4)
        ], p=0.8),
        transforms.RandomGrayscale(p=1.0),  # Ensure the image remains grayscale
        transforms.ToTensor(),
        normalize,
    ])
    
    paired_dataset = PairedDataset(root = opt.data_folder, transform=TwoCropTransform(train_transform), feature_transform=TwoCropTransformFeature())

    # Create data loader for paired dataset
    paired_loader = torch.utils.data.DataLoader(
        paired_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    return paired_loader

def set_model(opt):
    #model = SupConResNet(name=opt.model)
    opt.img_channels = 1
    opt.prior_dim = 8 * 32  # =256

    model = SupConResNet(
        name=opt.model,  # e.g. 'resnet18'
        img_in_channels=opt.img_channels,
        prior_dim=opt.prior_dim,
        feat_dim=64,
    )


    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(paired_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    #for idx, (images, labels) in enumerate(train_loader):
    for idx, (images, feats) in enumerate(paired_loader): ### image and prior information need to be matched in the train loader
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)  ## concat two versions of images
        feats = torch.cat([feats[0], feats[1]], dim=0)  ## do the same for features, check shape
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            feats = feats.cuda(non_blocking=True)
        bsz = images.size(0) // 2

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(paired_loader), optimizer)
        features = model(images, feats)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(paired_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader  # changed it to set_loader_as
    paired_loader = set_loader_as(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    log_file_path = os.path.join(opt.save_folder, 'training_log.txt')
    loss_file_path = os.path.join(opt.save_folder, 'training_loss.txt')

    # Open both files for the entire training loop
    with open(log_file_path, 'w') as log_file, open(loss_file_path, 'w') as loss_file:
        log_file.write("Epoch\tLoss\tLearning_Rate\n")
        loss_file.write("Epoch\tLoss\n")  # Add a header for the loss file

        # training routine (this loop must be inside the `with` block)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss = train(paired_loader, model, criterion, optimizer, epoch, opt)
            time2 = time.time()
            #print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
            print('epoch {}, loss {:.2f}, total time {:.2f}'.format(epoch, loss, time2 - time1))
            # Save loss and learning rate to the log file
            learning_rate = optimizer.param_groups[0]['lr']
            log_file.write(f"Epoch {epoch}\tLoss: {loss:.4f}\tLearning Rate: {learning_rate:.6f}\n")
            log_file.flush()  # Ensure the data is written to the file immediately

            # Save loss to the loss file
            loss_file.write(f"Epoch {epoch}\tLoss: {loss:.4f}\n")
            loss_file.flush()

            if epoch % opt.save_freq == 0:
                save_file = os.path.join(
                    opt.save_folder, f'ckpt_epoch_{epoch}.pth')
                save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()

