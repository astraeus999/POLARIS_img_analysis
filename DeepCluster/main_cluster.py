import argparse
import os
import time
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

import clustering
from util import AverageMeter, Logger, UnifLabelSampler
from resnet_big_old import SupConResNet, LinearClassifier

class UnsupervisedImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_paths = [
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if os.path.isfile(os.path.join(root, fname)) and fname.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)

# Define the getdataset function
def getdataset(batch_size, image_size=256, num_workers=4):
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to the desired image size
        transforms.ToTensor(),
    ])

    train_dataset = UnsupervisedImageDataset(root="./images", transform=preprocess)
    test_dataset = UnsupervisedImageDataset(root="./labeled_images", transform=preprocess)
    print(f"Number of training images: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')
    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                        default='Kmeans', help='clustering algorithm (default: Kmeans)')
    parser.add_argument('--nmb_cluster', '--k', type=int, default=2,
                        help='number of clusters for k-means (default: 2)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--reassign', type=float, default=1., help='how many epochs between reassignments of clusters (default: 1)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=16, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=500, help='iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='./ckpt/cluster_method_new_16', help='path to exp folder')
    parser.add_argument('--verbose', action='store_false', default=True, help='disable verbose mode (default: enabled)')
    return parser.parse_args()




def train(train_loader, model, classifier, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # Switch to training mode
    model.train()

    end = time.time()
    for i, (input_tensor, p_label) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move input to GPU
        input_tensor = input_tensor.cuda(non_blocking=True)
        p_label = p_label.cuda(non_blocking=True)  # <-- Add this line
        # print(p_label)
        # print(f'p_label shape: {p_label.shape}')
        # Compute output
        output = model(input_tensor)
        # print(output)
        # print(f'output shape: {output.shape}')
        result  = classifier(output)
        # print(result)
        # print(f'result shape: {result.shape}')
        # print(f'result shape: {result.shape}')
        # target = torch.zeros(output.size(0), dtype=torch.long).cuda()  # Dummy target for unsupervised learning
        # print(f'target shape: {target.shape}')
        loss = criterion(result, p_label)
        # print(f'loss: {loss.item()}')

        # Record loss
        losses.update(loss.item(), input_tensor.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg

def compute_features(dataloader, model, N, args):
    if args.verbose:
        print('Compute features')
    batch_time = AverageMeter()
    end = time.time()
    model.eval()
    with torch.no_grad():
        for i, (input_tensor, _) in enumerate(dataloader):
            input_tensor = input_tensor.cuda(non_blocking=True)
            aux = model(input_tensor).data.cpu().numpy()

            if i == 0:
                features = np.zeros((N, aux.shape[1]), dtype='float32')

            if i < len(dataloader) - 1:
                features[i * args.batch: (i + 1) * args.batch] = aux
            else:
                features[i * args.batch:] = aux

            batch_time.update(time.time() - end)
            end = time.time()

            # if args.verbose and (i % 200) == 0:
            #     print(f'{i} / {len(dataloader)}\tTime: {batch_time.val:.3f} ({batch_time.avg:.3f})')

    return features

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.verbose:
        print('Using Resnet18')
    # model = AutoEncoder().cuda()  # Initialize the AutoEncoder
    model = SupConResNet(
    name='resnet18', head='mlp', feat_dim=32
    ).cuda()
    classifier = LinearClassifier(name='resnet18', num_classes=2).cuda()  # Initialize the classifier
    cudnn.benchmark = True
    
    optimizer = torch.optim.SGD(
    list(model.parameters()) + list(classifier.parameters()),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=10**args.wd,
)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            print(checkpoint.keys())
            args.start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch: {args.start_epoch}")
            model.load_state_dict(checkpoint['state_dict'])
            classifier.load_state_dict(checkpoint['classifier_state_dict'])  # Load classifier state
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    os.makedirs(os.path.join(args.exp, 'checkpoints'), exist_ok=True)
    cluster_log = Logger(os.path.join(args.exp, 'clusters'))

    # Load datasets using the new getdataset function
    train_loader, test_loader = getdataset(batch_size=args.batch, image_size=256, num_workers=args.workers)

    deepcluster = clustering.__dict__[args.clustering](args.nmb_cluster)

    for epoch in range(args.start_epoch, args.epochs):
        end = time.time()

        features = compute_features(train_loader, model, len(train_loader.dataset), args)

        if args.verbose:
            print('Cluster the features')
        clustering_loss = deepcluster.cluster(features, verbose=args.verbose)

        if args.verbose:
            print('Assign pseudo labels')
        train_dataset = clustering.cluster_assign(deepcluster.images_lists, train_loader.dataset.image_paths)

        sampler = UnifLabelSampler(int(args.reassign * len(train_dataset)), deepcluster.images_lists)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=args.workers,
            sampler=sampler,
            pin_memory=True,
        )

        loss = train(train_dataloader, model, classifier, criterion, optimizer, epoch, args)

        if args.verbose:
            log_file_path = os.path.join(args.exp, 'training_log.txt')  # Define the log file path
            with open(log_file_path, 'a') as log_file:  # Open the file in append mode
                log_message = (f'Epoch [{epoch}] | Time: {time.time() - end:.7f}s | '
               f'Clustering Loss: {clustering_loss:.7f} | ResNet Loss: {loss:.7f}\n')
                print(log_message, end='')  # Print to console
                log_file.write(log_message)  # Write to log file

                try:
                    nmi = normalized_mutual_info_score(
                        clustering.arrange_clustering(deepcluster.images_lists),
                        clustering.arrange_clustering(cluster_log.data[-1])
                    )
                    # nmi_message = f'NMI against previous assignment: {nmi:.3f}\n'
                    # print(nmi_message, end='')  # Print to console
                    # log_file.write(nmi_message)  
                except IndexError:
                    pass

                log_file.write('####################### \n')  # Write separator to log file
                print('####################### \n') # Print separator to console    

        # torch.save({'epoch': epoch + 1,
        #             'arch': 'AutoEncoder',
        #             'state_dict': model.state_dict(),
        #             'optimizer': optimizer.state_dict()},
        #            os.path.join(args.exp, 'checkpoint.pth.tar'))
        if (epoch + 1) % 25 == 0:
            checkpoint_path = os.path.join(args.exp, f'ckpt_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'arch': 'AutoEncoder',
                'state_dict': model.state_dict(),
                'classifier_state_dict': classifier.state_dict(),  # Save classifier state
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            print(f"Checkpoint saved at: {checkpoint_path}")

        cluster_log.log(deepcluster.images_lists)

if __name__ == '__main__':
    args = parse_args()
    main(args)