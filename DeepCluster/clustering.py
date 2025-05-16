import time
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image and converts it to grayscale.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        a PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            path = dataset[idx]
            # print(path)
            pseudolabel = label_to_idx[pseudolabels[j]]
            images.append((path, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        path, pseudolabel = self.imgs[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca_dim=256):
    """Preprocess features: PCA-reduction and L2-normalization."""
    npdata = npdata.astype('float32')

    if pca_dim > 0 and pca_dim < npdata.shape[1]:
        pca = PCA(n_components=pca_dim, whiten=True)
        npdata = pca.fit_transform(npdata)

    npdata = normalize(npdata, norm='l2')
    return npdata


def run_kmeans_sklearn(x, nmb_clusters, verbose=False):
    """KMeans clustering using scikit-learn."""
    kmeans = KMeans(n_clusters=nmb_clusters, n_init=20, max_iter=300, random_state=None)
    kmeans.fit(x)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    
    if verbose:
        print(f"KMeans final inertia: {inertia:.4f}")
    
    return labels, inertia


class Kmeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering."""
        from time import time
        start = time()

        xb = preprocess_features(data)
        labels, loss = run_kmeans_sklearn(xb, self.k, verbose)

        self.images_lists = [[] for _ in range(self.k)]
        for idx, label in enumerate(labels):
            self.images_lists[label].append(idx)

        if verbose:
            print(f"KMeans clustering completed in {time() - start:.2f}s")
        return loss


def cluster_assign(images_lists, dataset):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    # print(f'pseudolabels:', pseudolabels)
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])  # For grayscale: 1 channel

    t = transforms.Compose([
    # transforms.Resize(256),                 # Resize to desired size
    # transforms.CenterCrop(224),             # Or use RandomCrop if desired
    transforms.RandomHorizontalFlip(),      # Optional: data augmentation
    transforms.ToTensor(),                  # Converts to tensor in [0,1]
    normalize                               # Normalize to [-1,1]
         ])
    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


