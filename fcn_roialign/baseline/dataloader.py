from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from pathlib import Path
import nibabel as nib
from torch.optim import lr_scheduler
import math
from skimage import measure


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# ROOT_DIR = Path('/Users/hariharan/PycharmProjects/SISS/')

ROOT_DIR = Path.cwd().parent.parent

def read_nib(path):
    'read from the path'
    img = nib.load(str(path))
    data = img.get_fdata()
    return np.array(data)


def read_single_scan(scan_slice, train = True):
    'read the first 4 CT scan slices and 1 mask'
    scan_idx, slice_idx = scan_slice
    scan_dir = ROOT_DIR / 'data' / ('train' if train else 'val') / str(scan_idx)
    scan_data = []
    paths = sorted([x for x in scan_dir.iterdir()])
    for path in paths:
        if path.is_dir():
            scan_type = path / (path.name + '.nii')
            slice = read_nib(scan_type)[:, :, slice_idx]
            scan_data.append(slice)

    return np.stack(scan_data, axis=-1)

def test_scan(scan_idx, train = True):
    scan_dir = ROOT_DIR / 'data' / ('train' if train else 'val') / str(scan_idx)
    paths = sorted([x for x in scan_dir.iterdir()])
    stack = read_nib(paths[0] / (paths[0].name + '.nii'))
    return stack.shape[-1]

class SISSDataset(Dataset):
    """SISS dataset."""

    def __init__(self, num_slices, num_scans, root_dir, train = True, transform=None):
        """
        Args:
            num_slices (int): 154 for number of slices
            num_scans (int): 3 scans available
            root_dir (string): Directory with all the NIB scan blobs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.num_slices = num_slices
        self.num_scans = num_scans
        self.total_samples = num_slices * num_scans
        self.sample_to_path = lambda x: (math.floor(x / num_slices) + 1, x % num_slices)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['background', 'lesion']

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        sample = read_single_scan(self.sample_to_path(idx), self.train)

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTupleTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # get the scans and the mask label.
        scans, label = sample[:, :, :-1], sample[:, :, -1]

        # get one hot representation for mask
        # label_onehot = (np.arange(2) == label[..., None]).astype(np.int64)

        #make datatypes consistent
        scans, label = scans.astype(np.float32), label.astype(np.int64)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        scans = scans.transpose((2, 0, 1))

        scans, label = torch.from_numpy(scans), torch.from_numpy(label)

        return scans, label



class ToRoIAlignTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # get the scans and the mask label.
        scans, label = sample[:, :, :-1], sample[:, :, -1]

        # get one hot representation for mask
        # label_onehot = (np.arange(2) == label[..., None]).astype(np.int64)

        #make datatypes consistent
        scans, label = scans.astype(np.float32), label.astype(np.float32)[np.newaxis,...]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        scans = scans.transpose((2, 0, 1))

        scans, label = torch.from_numpy(scans), torch.from_numpy(label)

        return scans, label



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(sample, (new_h, new_w))

        return img


class RandomRotate(object):
    """Rotate randomly the image in a sample."""
    def __init__(self, max_deg):
        self.max_deg = max_deg

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # generate a random angle between 0 to 20 degrees
        angle = np.random.uniform(0, 1) * self.max_deg

        # apply rotation
        sample = transform.rotate(sample, angle)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        h, w = sample.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample = sample[top: top + new_h,
                 left: left + new_w, :]

        return sample


class Normalize(object):
    """
    Normalise every slice of the image stack by apply 0-1 min max normalization.

    """
    def __init__(self):
        pass

    def normalizeSlice(self, image, newMin=0, newMax=1):
        '''
        Function to normalize a image using the min-max normalization

        Arguments:
            image {np.array} -- a numpy array of information

        Keyword Arguments:
            newMin {int} -- the new min value for normalization (default: {0})
            newMax {int} -- the new max value for normalization (default: {1})

        Returns:
            normalizedImage -- the normalized image as numpy array
        '''

        oldMin = np.nanmin(image)
        oldMax = np.nanmax(image)
        # Special case
        if oldMin == oldMax:
            # Case which we can make it as zero
            if oldMax <= 0:
                return np.zeros(shape=image.shape, dtype=np.float32)
            # Lost hope case
            print("[+] oldMin: %f, oldMax: %f" % (oldMin, oldMax))
            return image
        normalizedImage = ((image - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin

        return normalizedImage


    def __call__(self, sample):
        mean = [0.485, 0.456, 0.406, 0.449]
        std = [0.229, 0.224, 0.225, 0.226]

        for i in range(4):
            sample[:, :, i] = (self.normalizeSlice(sample[:, :, i]) - mean[i])/std[i]

        return sample


