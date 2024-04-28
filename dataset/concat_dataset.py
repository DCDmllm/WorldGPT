import torch
from .utils import mask_modality

class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
    
    def __getitem__(self, index):
        dataset_idx, idx, task_modality = index
        input, target = self.datasets[dataset_idx][idx]
        if 'path' in input:
            input['path'] = mask_modality(input['path'], task_modality['inputs'])
        if 'path' in target:
            target['path'] = mask_modality(target['path'], task_modality['targets'])
        if 'embeds' in input:
            input['embeds'] = mask_modality(input['embeds'], task_modality['inputs'])
        if 'embeds' in target:
            target['embeds'] = mask_modality(target['embeds'], task_modality['targets'])
        return input, target

class DecoderConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return super().__getitem__(index)
        dataset_idx, idx, _ = index
        return self.datasets[dataset_idx][idx]