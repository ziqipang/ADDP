import os
import json
import random
import numpy as np
from einops import rearrange
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    ToTensor, ToPILImage,
    ColorJitter, RandomAffine, RandomErasing
)


class RefSegDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        resolution: int = 256,
        return_all_sentences: bool = False,
        # augmentation
        intensity: float = 0.2,
        num_timesteps: int = 1000,
        sampling_group=10,
    ):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.split = split
        self.resolution = resolution
        self.return_all_sentences = return_all_sentences

        json_path = os.path.join(self.dataset_root, "anns", dataset_name, f"{split}.json")
        with open(json_path, "r") as f:
            self.samples = json.load(f)

        self.image_dir = os.path.join(self.dataset_root, "images", "train2014")
        self.mask_dir = os.path.join(self.dataset_root, "masks", dataset_name)

        # sampling weights
        self.intensity = intensity
        self.num_timesteps = num_timesteps
        assert num_timesteps % sampling_group == 0

        weights = torch.tensor([1.,  1.,  1.,  1.,  1.,  1.,  1., 2., 4., 95.])
        weights = weights.unsqueeze(-1).repeat(1, num_timesteps // sampling_group)
        weights = weights.flatten().contiguous()
        self.weights = weights / weights.sum()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = os.path.join(self.image_dir, sample["img_name"])
        mask_path = os.path.join(self.mask_dir, f"{sample['segment_id']}.png")
        prompts = [sent["sent"] for sent in sample["sentences"]]

        if self.return_all_sentences:
            prompt = prompts
        else:
            random.shuffle(prompts)
            prompt = prompts[0]

        image = Image.open(image_path).convert("RGB")
        mask_bin = Image.open(mask_path).convert("L")

        # resize
        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
        mask_bin = mask_bin.resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        image = np.array(image)
        mask_bin = np.array(mask_bin)[..., np.newaxis] / 255

        edit_target = np.zeros_like(image)
        edit_target[:, :, 0] = 255  # red
        tgt_image = (1 - mask_bin) * image + mask_bin * edit_target

        if self.split == "train":  # augmentation
            t = random.choices(range(self.num_timesteps), self.weights, k=1)[0]
            intensity = self.intensity * (t / self.num_timesteps)

            self.color_transform = Compose([
                ToPILImage(),
                ColorJitter(brightness=intensity, contrast=intensity, saturation=intensity),
            ])

            self.mask_transform = Compose([
                ToTensor(),
                RandomAffine(degrees=50*intensity, translate=(intensity, intensity), scale=(1-intensity, 1+intensity)),
                RandomErasing(p=0.5, scale=(0, intensity), ratio=(0.5, 2.0), value=0),
            ])

            edit_target_aug = self.color_transform(edit_target)
            mask_bin_aug = self.mask_transform(mask_bin)[0]
            mask_bin_aug = np.array(mask_bin_aug)[..., np.newaxis]
            tgt_image_aug = (1 - mask_bin_aug) * image + mask_bin_aug * edit_target_aug
        else:  # no augmentation
            t = 0
            tgt_image_aug = tgt_image

        # h * w * c --> c * h * w
        image = rearrange(2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w")
        tgt_image = rearrange(2 * torch.tensor(np.array(tgt_image)).float() / 255 - 1, "h w c -> c h w")
        tgt_image_aug = rearrange(2 * torch.tensor(np.array(tgt_image_aug)).float() / 255 - 1, "h w c -> c h w")
        return dict(edited=tgt_image, edit=dict(c_concat=image, c_crossattn=prompt), edited_aug=tgt_image_aug, t=t)


# sample example: {
#   'bbox': [103, 299, 237, 476],
#   'cat': 0, 'segment_id': 0, 'img_name': 'COCO_train2014_000000581857.jpg',
#   'sentences': [
#     { 'idx': 0, 'sent_id': 0, 'sent': 'the lady with the blue shirt' },
#     { 'idx': 1, 'sent_id': 1, 'sent': 'lady with back to us' },
#     { 'idx': 2, 'sent_id': 2, 'sent': 'blue shirt' },
#   ],
#   'sentences_num': 3,
# }
