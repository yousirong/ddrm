"""
BUSI (Breast Ultrasound Images) Dataset Loader for AV-DRUS
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class BUSIDataset(Dataset):
    """
    BUSI Dataset for ultrasound image reconstruction
    """
    
    def __init__(self, root, split='train', transform=None, image_size=256, use_masks=False):
        """
        Args:
            root: path to BUSI dataset (should contain benign, malignant, normal folders)
            split: 'train', 'val', or 'test'
            transform: transform to apply to images
            image_size: target image size
            use_masks: whether to load segmentation masks
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.use_masks = use_masks
        
        # Basic transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # Load image paths
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        
        categories = ['benign', 'malignant', 'normal']
        category_to_label = {'benign': 0, 'malignant': 1, 'normal': 2}
        
        for category in categories:
            category_path = os.path.join(root, category)
            if os.path.exists(category_path):
                # Get all PNG files that are not masks
                image_files = glob.glob(os.path.join(category_path, "*.png"))
                image_files = [f for f in image_files if not f.endswith('_mask.png')]
                
                # Split data
                np.random.seed(42)  # For reproducible splits
                np.random.shuffle(image_files)
                
                n_total = len(image_files)
                n_train = int(0.7 * n_total)
                n_val = int(0.15 * n_total)
                
                if split == 'train':
                    selected_files = image_files[:n_train]
                elif split == 'val':
                    selected_files = image_files[n_train:n_train+n_val]
                else:  # test
                    selected_files = image_files[n_train+n_val:]
                
                for img_path in selected_files:
                    self.image_paths.append(img_path)
                    self.labels.append(category_to_label[category])
                    
                    # Check for corresponding mask
                    if self.use_masks:
                        mask_path = img_path.replace('.png', '_mask.png')
                        if os.path.exists(mask_path):
                            self.mask_paths.append(mask_path)
                        else:
                            self.mask_paths.append(None)
        
        print(f"BUSI Dataset [{split}]: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Convert grayscale to RGB for compatibility with pretrained models
        image = image.convert('RGB')
        
        # Apply base transforms
        image = self.base_transform(image)
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Load mask if requested
        if self.use_masks and len(self.mask_paths) > idx and self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
            mask = self.base_transform(mask)
            return image, self.labels[idx], mask
        
        return image, self.labels[idx]


class BUSIUltrasoundSimulation(Dataset):
    """
    BUSI dataset with simulated ultrasound RF data and degradation
    """
    
    def __init__(self, root, split='train', transform=None, image_size=256, 
                 noise_level=0.1, speckle_variance=0.2):
        """
        Args:
            root: path to BUSI dataset
            split: train/val/test split
            transform: additional transforms
            image_size: target image size
            noise_level: amount of additive noise
            speckle_variance: variance for multiplicative speckle noise
        """
        self.base_dataset = BUSIDataset(root, split, None, image_size)
        self.transform = transform
        self.noise_level = noise_level
        self.speckle_variance = speckle_variance
        
        # For simulating RF data
        self.num_elements = 128
        self.num_samples = 512
        
    def __len__(self):
        return len(self.base_dataset)
    
    def simulate_rf_data(self, image):
        """
        Simulate RF channel data from ground truth image
        """
        B, C, H, W = image.shape if len(image.shape) == 4 else (1, *image.shape)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Simple simulation: create RF data by adding spatial and temporal structure
        rf_data = torch.randn(B, self.num_elements, H * W // self.num_elements)
        
        # Add image content to RF data (simplified)
        image_flat = image.mean(dim=1).view(B, -1)  # Average over channels
        if image_flat.shape[1] != rf_data.shape[2]:
            image_flat = torch.nn.functional.interpolate(
                image_flat.unsqueeze(1), size=rf_data.shape[2], mode='linear'
            ).squeeze(1)
        
        # Modulate RF data with image content
        rf_data = rf_data + 0.5 * image_flat.unsqueeze(1).expand_as(rf_data)
        
        # Add noise
        rf_data = rf_data + self.noise_level * torch.randn_like(rf_data)
        
        return rf_data.squeeze(0) if B == 1 else rf_data
    
    def simulate_das_beamforming(self, rf_data):
        """
        Simulate basic DAS beamforming
        """
        if len(rf_data.shape) == 2:
            rf_data = rf_data.unsqueeze(0)
        
        # Simple DAS: average across elements
        das_image = rf_data.mean(dim=1)  # [B, num_samples]
        
        # Reshape to 2D image
        B, L = das_image.shape
        H = W = int(np.sqrt(L))
        if H * W != L:
            # Pad to nearest square
            target_size = int(np.ceil(np.sqrt(L)))
            das_image = torch.nn.functional.pad(das_image, (0, target_size**2 - L))
            H = W = target_size
        
        das_image = das_image.view(B, 1, H, W)
        
        # Resize to target image size
        das_image = torch.nn.functional.interpolate(
            das_image, size=(self.base_dataset.image_size, self.base_dataset.image_size),
            mode='bilinear', align_corners=False
        )
        
        # Add speckle noise (multiplicative)
        speckle = 1 + self.speckle_variance * torch.randn_like(das_image)
        das_image = das_image * speckle
        
        # Expand to 3 channels for compatibility
        das_image = das_image.expand(-1, 3, -1, -1)
        
        return das_image.squeeze(0) if das_image.shape[0] == 1 else das_image
    
    def __getitem__(self, idx):
        # Get base image and label
        image, label = self.base_dataset[idx]
        
        # Simulate RF data and DAS reconstruction
        rf_data = self.simulate_rf_data(image)
        das_image = self.simulate_das_beamforming(rf_data)
        
        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)
            das_image = self.transform(das_image)
        
        return {
            'clean_image': image,
            'das_image': das_image,
            'rf_data': rf_data,
            'label': label
        }