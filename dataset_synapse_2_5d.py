"""
2.5D Dataset - FIXED:
  1. image output shape [n_slices*3, H, W] to match model in_chans
  2. Robust __getitem__ - image/label always assigned before use
  3. h5py file closed properly with context manager
  4. _get_adjacent_slices handles missing center slice gracefully
"""

import os
import random
import h5py
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


class Synapse_dataset_2_5D(Dataset):
    """
    2.5D Dataset for Synapse.
    Each training sample returns:
        image: [n_slices * 3, H, W]  — each slice replicated to 3 channels
        label: [H, W]
    """

    def __init__(self, base_dir, list_dir, split='train', n_slices=3, transform=None):
        self.transform = transform
        self.split     = split
        self.n_slices  = n_slices
        self.pad       = n_slices // 2
        self.data_dir  = base_dir

        if split not in ('train', 'test_vol'):
            raise ValueError(f"split must be 'train' or 'test_vol', got '{split}'")

        list_file = os.path.join(list_dir, 'train.txt' if split == 'train' else 'test_vol.txt')
        with open(list_file, 'r') as f:
            self.sample_list = [line.strip() for line in f if line.strip()]

        print(f"✓ Loaded {len(self.sample_list)} samples from {list_file}")

        if split == 'train':
            self._build_case_mapping()

    def _build_case_mapping(self):
        self.case_to_slices = {}
        for name in self.sample_list:
            parts = name.split('_slice')
            if len(parts) == 2:
                case = parts[0]
                sidx = int(parts[1])
                self.case_to_slices.setdefault(case, []).append(sidx)
        for k in self.case_to_slices:
            self.case_to_slices[k].sort()
        print(f"✓ Found {len(self.case_to_slices)} cases")

    def _load_slice(self, case_name, slice_idx):
        path = os.path.join(self.data_dir, f"{case_name}_slice{slice_idx:03d}.npz")
        if os.path.exists(path):
            data = np.load(path)
            return data['image'], data['label']
        return None, None

    def _get_adjacent_slices(self, case_name, center_idx):
        """Build [n_slices*3, H, W] stack. Raises if center slice missing."""
        if case_name in self.case_to_slices:
            valid = self.case_to_slices[case_name]
            lo, hi = min(valid), max(valid)
        else:
            lo = center_idx - self.pad
            hi = center_idx + self.pad

        center_img, center_lbl = self._load_slice(case_name, center_idx)
        if center_img is None:
            raise RuntimeError(
                f"Center slice not found: {case_name}_slice{center_idx:03d}.npz"
            )

        slices = []
        for offset in range(-self.pad, self.pad + 1):
            idx = int(np.clip(center_idx + offset, lo, hi))
            img, _ = self._load_slice(case_name, idx)
            if img is None:
                img = center_img
            slices.append(img)

        rgb_slices  = [np.stack([s, s, s], axis=0) for s in slices]
        image_stack = np.concatenate(rgb_slices, axis=0)   # [n_slices*3, H, W]
        return image_stack, center_lbl

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]

        if self.split == 'train':
            parts = sample_name.split('_slice')
            if len(parts) != 2:
                raise ValueError(f"Invalid train sample name: '{sample_name}'")
            case_name = parts[0]
            slice_idx = int(parts[1])
            image, label = self._get_adjacent_slices(case_name, slice_idx)

        else:  # test_vol
            filepath = os.path.join(self.data_dir, f"{sample_name}.npy.h5")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"H5 file not found: {filepath}")
            with h5py.File(filepath, 'r') as f:
                image = f['image'][:]
                label = f['label'][:]

        sample = {'image': image, 'label': label, 'case_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomGenerator_2_5D(object):
    """Data augmentation for 2.5D. Expects image [n_slices*3, H, W], label [H, W]."""

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']   # [C, H, W]
        label = sample['label']   # [H, W]
        C, x, y = image.shape

        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()

        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()

        if random.random() > 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            label = np.rot90(label, k).copy()

        if x != self.output_size[0] or y != self.output_size[1]:
            scale   = (self.output_size[0] / x, self.output_size[1] / y)
            resized = [zoom(image[c], scale, order=3) for c in range(C)]
            image   = np.stack(resized, axis=0)
            label   = zoom(label, scale, order=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32)).long()

        return {
            'image':     image,
            'label':     label,
            'case_name': sample.get('case_name', '')
        }
