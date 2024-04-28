import os
import argparse
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model.preprocessor import PreProcessor

def parser_args():
    parser = argparse.ArgumentParser(description='preprocess parameters')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp32', 'fp16', 'bf16'])

    parser.add_argument('--modality', type=str, required=True, choices=['image', 'video', 'audio'])
    parser.add_argument('--languagebind_path', type=str, required=True)
    return parser.parse_args()


class MyDataset(Dataset):
    def __init__(self, root, modality):
        self.root = root
        self.modality = modality
        self.names = os.listdir(self.root)
    
    def __getitem__(self, index):
        return {self.modality: os.path.join(self.root, self.names[index])}
    
    def __len__(self):
        return len(self.names)


suffix = {
    'image': '.jpg',
    'video': '.mp4',
    'audio': '.wav'
}


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)

    if args['output_root'] is None:
        args['output_root'] = os.path.join(os.path.dirname(args['data_root']), 'Train_pt')
    os.makedirs(args['output_root'])
    
    if args['dtype'] == 'fp16':
        args['dtype'] = torch.float16
    elif args['dtype'] == 'bf16':
        args['dtype'] = torch.bfloat16
    else:
        args['dtype'] = torch.float

    dataset = MyDataset(args['data_root'], args['modality'])
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    preprocessor = PreProcessor({
        'preprocess_modality': [args['modality']],
        'languagebind_path': {args['modality']: args['languagebind_path']}
    })
    preprocessor.to_('cuda', dtype=args['dtype'])

    for batch in tqdm(dataloader):
        with torch.no_grad():
            outputs, _ = preprocessor(batch)
        outputs = outputs['image'].cpu().float()
        for data_path, embeds in zip(batch['image'], outputs):
            name = os.path.basename(data_path).replace(suffix[args['modality']], '.pt')
            torch.save(embeds, os.path.join(args['output_root'], name))
