# vision_dataset/load_dataset.py

import io
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
from datasets import load_dataset, interleave_datasets
from huggingface_hub import login

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


class ImagenetWDSIterableDataset(IterableDataset):
    """
    Streaming IterableDataset for timm/imagenet-1k-wds (parquet shards).
    Each record has 'image__value' (bytes) and 'label' (int).
    """
    def __init__(self, split="train", image_size=224):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.transform_train = get_imagenet_transforms(image_size, is_training=True)
        self.transform_val   = get_imagenet_transforms(image_size, is_training=False)

        # Load the parquet-based WebDataset, streaming mode
        self.dataset = load_dataset(
            "timm/imagenet-1k-wds",
            split=split,
            streaming=True,
            # use_auth_token=True,       # if needed
            cache_dir="hf_cache",      # adjust if desired
        )

    def __iter__(self):
        for example in self.dataset:
            # Extract raw image object (could be PIL or bytes or array)
            img_obj = example.get("jpg", None) or example.get("image")
            
            # Convert bytes->PIL if needed
            if isinstance(img_obj, (bytes, bytearray)):
                img = Image.open(io.BytesIO(img_obj))
            elif isinstance(img_obj, Image.Image):
                img = img_obj
            else:
                # e.g. NumPy array
                img = Image.fromarray(img_obj)
            
            # Force RGB mode
            img = img.convert("RGB")
            
            # Now safe to apply ViT transforms
            if self.split == "train":
                tensor = self.transform_train(img)
            else:
                tensor = self.transform_val(img)
            
            label = example["json"]["label"]
            yield {
                "pixel_values": tensor,
                "labels": torch.tensor(label, dtype=torch.long),
            }


def load_vision_dataset_from_config(cfg):
    """
    Returns:
      train_dataset: IterableDataset for training
      val_dataset: IterableDataset for validation
    """
    train_ds = ImagenetWDSIterableDataset(split="train", image_size=cfg.image_size)
    val_ds   = ImagenetWDSIterableDataset(split="validation", image_size=cfg.image_size)
    return train_ds, val_ds


# Example DataLoader usage:
# train_ds, val_ds = load_vision_dataset_from_config(cfg)
# train_loader = DataLoader(train_ds, batch_size=cfg.per_device_train_batch_size,
#                           num_workers=cfg.dataloader_num_workers)
