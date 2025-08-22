# vision_dataset/load_dataset.py

import io
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, interleave_datasets
from huggingface_hub import login
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor

login("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")  # replace with your actual token


def get_imagenet_transforms(image_size=224, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225]),
        ])


class OptimizedImagenetWDS(IterableDataset):
    def __init__(self, split="train", image_size=224, num_workers=4):
        super().__init__()
        self.split = split
        # Use torchvision v2 transforms (faster)
        self.transform = transforms_v2.Compose([
            transforms_v2.ToImage(),
            transforms_v2.ToDtype(torch.uint8, scale=True),
            transforms_v2.RandomResizedCrop(image_size) if split == "train" 
                else transforms_v2.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            transforms_v2.CenterCrop(image_size) if split == "validation" else transforms_v2.Identity(),
            transforms_v2.RandomHorizontalFlip() if split == "train" else transforms_v2.Identity(),
            transforms_v2.ToDtype(torch.float32, scale=True),
            transforms_v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Enable faster data loading
        self.dataset = load_dataset(
            "timm/imagenet-1k-wds",
            split=split,
            streaming=True,
            cache_dir="hf_cache",
        ).with_format("torch")  # Use torch format directly
    
    def __iter__(self):
        for example in self.dataset:
            # Direct tensor conversion - much faster
            img_tensor = example["jpg"]  # Already a tensor if using with_format("torch")
            tensor = self.transform(img_tensor)
            yield {
                "pixel_values": tensor,
                "labels": torch.tensor(example["json"]["label"], dtype=torch.long),
            }



def load_vision_dataset_from_config(cfg):
    """
    Returns:
      train_dataset: IterableDataset for training
      val_dataset: IterableDataset for validation
    """
    train_ds = OptimizedImagenetWDS(split="train", image_size=cfg.image_size)
    val_ds   = OptimizedImagenetWDS(split="validation", image_size=cfg.image_size)
    return train_ds, val_ds


# Example DataLoader usage:
# train_ds, val_ds = load_vision_dataset_from_config(cfg)
# train_loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size,
#                           num_workers=cfg.dataloader_num_workers)
