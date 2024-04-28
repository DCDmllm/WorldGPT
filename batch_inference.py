import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.worldgpt import load_worldgpt_model
from dataset.utils import to, mask_modality, serialize_task
from config import load_config


class StateDataset(Dataset):
    def __init__(self, metadata):
        self.root = metadata['root']
        self.modality = metadata['modality']
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
        input['text'] = ann['action']['text']
        for m in self.modality:
            if ann['state0'][m] != '':
                input[m] = self._load_languagebind(m, ann['state0'][m])
            target[m] = ann['state1'][m]
        return input, target

    def __len__(self):
        return len(self.annotation)


def collate(data):
    input_list = [d[0] for d in data]
    target_list = [d[1] for d in data]
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


if __name__ == '__main__':
    _args = {'mode': 'test', 'cfg_path': 'config/batch_inference.yaml'}
    args = load_config(_args)
    args.update(_args)

    if 'dtype' not in args:
        args['dtype'] = torch.float16
    elif args['dtype'] == 'fp16':
        args['dtype'] = torch.float16
    elif args['dtype'] == 'bf16':
        args['dtype'] = torch.bfloat16
    else:
        args['dtype'] = torch.float

    model = load_worldgpt_model(**args)
    model = model.eval().to(device='cuda', dtype=args['dtype'])


    for metadata in args['dataset_list']:
        dataset = StateDataset(metadata)
        dataloader = DataLoader(dataset, batch_size=args['batch_size'], num_workers=4, pin_memory=True, shuffle=False, collate_fn=collate)
        for inputs, targets in tqdm(dataloader):
            for task in args['modality_modes']:
                task_name = serialize_task(task)
                for m in task['targets']:
                    os.makedirs(os.path.join(metadata['log_path'], task_name, m))
                masked_inputs = mask_modality(inputs, return_modality=task['inputs'])
                outputs = model.generate(masked_inputs, task['targets'], max_tgt_length=args['max_tgt_length'],  # (B, 768)
                                        top_p=args['top_p'], temperature=args['temperature'])
                outputs = to(outputs, device='cpu', dtype=torch.float)
                for m in outputs:
                    for embeds, name in zip(outputs[m], targets[m]):
                        torch.save(embeds, os.path.join(metadata['log_path'], task_name, m, name.replace(dataset.suffix[m], '.pt')))
