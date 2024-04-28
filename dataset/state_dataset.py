import os 
import json

import torch
from torch.utils.data import Dataset


class StateDataset(Dataset):
    def __init__(self, metadata, return_path=True, return_embeds=True):
        self.root = metadata['root']
        self.modality = metadata['modality']
        self.return_path = return_path
        self.return_embeds = return_embeds
        with open(metadata['annotaion_path'], 'r') as f:
            self.annotation = json.load(f)

        self.suffix = {'image': '.jpg', 'video': '.mp4', 'audio': '.wav'}

    def _load_languagebind(self, modality, name):
        if name == '':
            raise ValueError(f'Dataset {self.root} missing value in modality {modality}')
        name = name.replace(self.suffix[modality], '.pt')
        path = os.path.join(self.root, modality, 'Train_pt', name)
        return torch.load(path, map_location=torch.device('cpu'))

    def __getitem__(self, index):
        input, target = {}, {}
        ann = self.annotation[index]
        if self.return_path:
            input['path'], target['path'] = {}, {}
            input['path']['text'] = ann['action']['text']
            for m in self.modality:
                if ann['state0'][m] != '':
                    input['path'][m] = os.path.join(self.root, m, 'Train', ann['state0'][m])
                if ann['state1'][m] != '':
                    target['path'][m] = os.path.join(self.root, m, 'Train', ann['state1'][m])
        if self.return_embeds:
            input['embeds'], target['embeds'] = {}, {}
            input['embeds']['text'] = ann['action']['text']
            for m in self.modality:
                if ann['state0'][m] != '':
                    input['embeds'][m] = self._load_languagebind(m, ann['state0'][m])
                if ann['state1'][m] != '':
                    target['embeds'][m] = self._load_languagebind(m, ann['state1'][m])
        return input, target

    def __len__(self):
        return len(self.annotation)


class CaptionDataset(Dataset):
    def __init__(self, metadata):
        self.root = metadata['root']
        self.modality = metadata['modality']

        files = os.listdir(self.root)
        files.sort()
        self.id_list = [name.split('.')[0] for name in files[::3]]

        self.suffix = {'image': '.jpg', 'video': '.mp4', 'audio': '.wav'}

    def __getitem__(self, index):
        input, target = {}, {}
        id = self.id_list[index]
        with open(os.path.join(self.root, f'{id}.txt'), 'r') as f:
            caption = f.read()
        input['path'], target['path'] = {}, {}
        input['path']['text'] = caption
        for m in self.modality:
            target['path'][m] = os.path.join(self.root, id + self.suffix[m])

        input['embeds'] = input['path']
        target['embeds'] = target['path']

        return input, target

    def __len__(self):
        return len(self.id_list)


class EvalStateDataset(Dataset):
    def __init__(self, eval_ann, dataset_roots, return_path=True, return_embeds=True):
        with open(eval_ann, 'r') as f:
            # self.annotation = json.load(f)['datas']
            self.annotation = json.load(f)
        self.dataset_roots = dataset_roots
        self.return_path = return_path
        self.return_embeds = return_embeds
        self.suffix = {'image': '.jpg', 'video': '.mp4', 'audio': '.wav'}
    
    def _load_languagebind(self, dataset_name, modality, name):
        if name == '':
            raise ValueError(f'Dataset {dataset_name} missing value in modality {modality}')
        name = name.replace(self.suffix[modality], '.pt')
        path = os.path.join(self.dataset_roots[dataset_name], modality, 'Train_pt', name)
        return torch.load(path, map_location=torch.device('cpu'))

    def __getitem__(self, index):
        input, target = {}, {}
        ann = self.annotation[index]
        dataset_name = ann['origin']

        if self.return_path:
            input['path'], target['path'] = {}, {}
            input['path']['text'] = ann['action']['text']
            for m in ['image', 'video', 'audio']:
                if ann['state0'][m] != '':
                    input['path'][m] = os.path.join(self.dataset_roots[dataset_name], m, 'Train', ann['state0'][m])
                if ann['state1'][m] != '':
                    target['path'][m] = os.path.join(self.dataset_roots[dataset_name], m, 'Train', ann['state1'][m])
        
        if self.return_embeds:
            input['embeds'], target['embeds'] = {}, {}
            input['embeds']['text'] = ann['action']['text']
            for m in ['image', 'video', 'audio']:
                if ann['state0'][m] != '':
                    input['embeds'][m] = self._load_languagebind(dataset_name, m, ann['state0'][m])
                if ann['state1'][m] != '':
                    target['embeds'][m] = self._load_languagebind(dataset_name, m, ann['state1'][m])
        
        return input, target

    def __len__(self):
        return len(self.annotation)


def collate_embeds(data):
    input_list = [d[0]['embeds'] for d in data]
    target_list = [d[1]['embeds'] for d in data]
    inputs, targets = {}, {}
    
    inputs['text'] = [i['text'] for i in input_list]
    for m in input_list[0]:
        if m == 'text':
            continue
        inputs[m] = [i[m] for i in input_list]
        if isinstance(inputs[m][0], torch.Tensor):
            inputs[m] = torch.stack(inputs[m])
    for m in target_list[0]:
        targets[m] = [t[m] for t in target_list]
        if isinstance(targets[m][0], torch.Tensor):
            targets[m] = torch.stack(targets[m])
    
    return inputs, targets

def collate_path(data):
    input_list = [d[0]['path'] for d in data]
    target_list = [d[1]['path'] for d in data]
    inputs, targets = {}, {}

    for m in input_list[0]:
        inputs[m] = [i[m] for i in input_list]
    for m in target_list[0]:
        targets[m] = [t[m] for t in target_list]
    
    return inputs, targets

def collate_all(data):
    inputs_path, targets_path = collate_path(data)
    inputs_embeds, targets_embeds = collate_embeds(data)
    return {'path': inputs_path, 'embeds': inputs_embeds}, {'path': targets_path, 'embeds': targets_embeds}