import os
import pickle
import numpy as np
import torch
from torch.utils.data import Sampler

import models


def load_model(path: str):
    """Loads a model checkpoint and returns the model (DataParallel-free)."""
    if os.path.isfile(path):
        print(f"=> loading checkpoint '{path}'")
        checkpoint = torch.load(path, map_location='cpu')

        # Determine output dimension from top layer
        N = checkpoint['state_dict']['top_layer.bias'].size(0)

        # Detect if Sobel filter was used
        sobel = 'sobel.0.weight' in checkpoint['state_dict']

        # Create model instance
        model = models.__dict__[checkpoint['arch']](sobel=sobel, out=N)

        # Remove 'module.' prefix if needed (from DataParallel)
        new_state_dict = {
            key.replace('module.', ''): val
            for key, val in checkpoint['state_dict'].items()
        }
        model.load_state_dict(new_state_dict)
        print("=> model loaded.")
        return model
    else:
        print(f"=> no checkpoint found at '{path}'")
        return None


class UnifLabelSampler(Sampler):
    """Samples elements uniformly across pseudo-labels."""

    def __init__(self, N: int, images_lists: dict):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        non_empty_clusters = [v for v in self.images_lists if len(v) > 0]
        nmb_non_empty_clusters = len(non_empty_clusters)
        size_per_label = self.N // nmb_non_empty_clusters + 1

        indexes = []
        for i in range(len(self.images_lists)):
            cluster = self.images_lists[i]
            if not cluster:
                continue
            sampled = np.random.choice(
                cluster,
                size=size_per_label,
                replace=(len(cluster) <= size_per_label)
            )
            indexes.extend(sampled)

        np.random.shuffle(indexes)
        indexes = list(map(int, indexes))
        return indexes[:self.N] if len(indexes) >= self.N else indexes + indexes[:self.N - len(indexes)]

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t: int, lr_0: float):
    """Applies square root learning rate decay."""
    for param_group in optimizer.param_groups:
        wd = param_group.get('weight_decay', 0.0)
        param_group['lr'] = lr_0 / np.sqrt(1 + lr_0 * wd * t)


class Logger:
    """Tracks and logs metrics at each epoch to disk."""

    def __init__(self, path: str):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(self.path, 'wb') as f:
            pickle.dump(self.data, f, protocol=pickle.HIGHEST_PROTOCOL)
