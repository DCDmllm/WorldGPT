import os 
import json

import torch
from torch.utils.data import Dataset
from PIL import Image


class DecoderStateDataset(Dataset):
    def __init__(self, metadata, train_modality, state0_transforms=None, state1_transforms=None):
        self.root = metadata['root']
        self.modality = train_modality
        assert self.modality in metadata['modality']
        if self.modality == 'image':
            self.postfix = '.jpg'
        elif self.modality == 'video':
            self.postfix = '.mp4'
        elif self.modality == 'audio':
            self.postfix = '.wav'
        else:
            raise ValueError(f'Wrong modality {self.modality}')
        with open(metadata['annotaion_path'], 'r') as f:
            self.annotation = json.load(f)
        self.state0_transforms = state0_transforms
        self.state1_transforms = state1_transforms
    
    def _load_raw(self, name):
        path = os.path.join(self.root, self.modality, 'Train', name)
        if self.modality == 'image':
            return Image.open(path)
        else:
            raise ValueError(f'Wrong modality {self.modality}')

    def _load_languagebind(self, name):
        if name == '':
            raise ValueError(f'Dataset {self.root} missing value in modality {self.modality}')
        path = os.path.join(self.root, self.modality, 'Train_pt', name)
        return torch.load(path, map_location=torch.device('cpu'))

    def __getitem__(self, index):
        ann = self.annotation[index]
        state0_raw = self._load_raw(ann['state0'][self.modality])
        state1_embeds = self._load_languagebind(ann['state1'][self.modality].replace(self.postfix, '.pt'))
        state1_raw = self._load_raw(ann['state1'][self.modality])

        if self.state0_transforms:
            state0_raw = self.state0_transforms(state0_raw)
        if self.state1_transforms:
            state1_raw = self.state1_transforms(state1_raw)
        
        return state0_raw, state1_embeds, state1_raw

    def __len__(self):
        return len(self.annotation)


def collate_decoder(data):
    batch_state0_raw, batch_state1_embeds, batch_state1_raw = [], [], []
    for state0_raw, state1_embeds, state1_raw in data:
        batch_state0_raw.append(state0_raw)
        batch_state1_embeds.append(state1_embeds)
        batch_state1_raw.append(state1_raw)
    
    batch_state1_embeds = torch.stack(batch_state1_embeds)
    if isinstance(batch_state0_raw[0], torch.Tensor):
        batch_state0_raw = torch.stack(batch_state0_raw)
    if isinstance(batch_state1_raw[0], torch.Tensor):
        batch_state1_raw = torch.stack(batch_state1_raw)

    return {
        "pixel_values": batch_state1_raw,
        "conditioning_pixel_values": batch_state0_raw,
        "input_embeds": batch_state1_embeds
    }