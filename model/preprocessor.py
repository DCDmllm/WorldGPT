import os
from PIL import Image
import numpy as np
import torch

from transformers import CLIPTextModel, CLIPTokenizer
from .languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer


class PreProcessor:
    def __init__(self, args):
        self.device = torch.cuda.current_device()
        if 'preprocess_modality' in args:
            self.modality = args['preprocess_modality']
        else:
            self.modality = ['image']
        self.dtype = torch.float
        
        print(f'Loading LanguageBind model...')
        self.model = LanguageBind(clip_type={m: args['languagebind_path'][m] for m in self.modality}, cache_dir='./cache_dir')
        self.model = self.model.to(self.device).eval()
        self.processors = {m: transform_dict[m](self.model.modality_config[m]) for m in self.modality}
        self.tokenizer = LanguageBindImageTokenizer.from_pretrained(args['languagebind_path']['image'], cache_dir='./cache_dir/tokenizer_cache_dir')
    
    def __call__(self, batch):
        inputs = {}
        if len(batch) == 1 and 'text' in batch: # text only inputs
            inputs['language'] = self.tokenizer(batch['text'], max_length=77, padding='max_length', truncation=True, return_tensors='pt')
            inputs['language'] = to_device(inputs['language'], self.device)
        else:
            for m in self.modality:
                if m in batch:
                    inputs[m] = to_device(self.processors[m](batch[m]), self.device)
                for k in inputs[m]:
                    inputs[m][k] = inputs[m][k].to(dtype=self.dtype)
        with torch.no_grad():
            model_outputs = self.model(inputs)
        outputs = {}
        for m in batch:
            if m in self.modality:
                outputs[m] = model_outputs[m].detach()
            elif m == 'text':
                outputs[m] = batch['text']
        return outputs, model_outputs.get('language')
    
    def to_(self, device=None, dtype=None):
        self.model = self.model.to(device=device, dtype=dtype)
        if dtype:
            self.dtype = dtype