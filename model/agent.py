import os
import json
import types
import datetime
import logging
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import pandas as pd
from tqdm import tqdm

from config import load_config
from dataset import load_dataset
from dataset.utils import to, mask_modality, serialize_task


class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.eval_args = deepcopy(self.args)
        self.eval_args['mode'] = 'validation'
        eval_args = load_config(self.eval_args)
        self.eval_args.update(eval_args)

        self.print_model_parameters()
        self.writer = SummaryWriter(args['log_path'])

        if self.args['load_path']:
            self.load_parameters(self.args['load_path'])
        
        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(
            self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, inputs, target_modality):
        self.ds_engine.module.eval()
        output = self.ds_engine.generate(inputs, target_modality)
        return output

    @torch.no_grad()
    def eval(self, current_step):
        self.ds_engine.module.eval()

        _, eval_dataloader = load_dataset(self.eval_args, self.eval_args['dataset_list'])
        sim = {}
        for m in ['image', 'video', 'audio']:
            sim[m] = {}
            for task in self.eval_args['modality_modes']:
                if m in task['targets']:
                    sim[m][serialize_task(task)] = []
        for inputs, targets in tqdm(eval_dataloader, desc='Validation samples'):
            for task in self.eval_args['modality_modes']:
                masked_inputs = mask_modality(inputs, return_modality=task['inputs'])
                outputs = self.ds_engine.generate(masked_inputs, task['targets'], max_tgt_length=self.eval_args['max_tgt_length'],  # (B, 768)
                                                  top_p=self.eval_args['top_p'], temperature=self.eval_args['temperature'])
                outputs = to(outputs, device='cpu', dtype=torch.float)
                for m in outputs:
                    sim[m][serialize_task(task)].append(F.cosine_similarity(outputs[m], targets[m], dim=1))  # (B,)
        
        for m in ['image', 'video', 'audio']:
            for task in sim[m]:
                sim[m][task] = torch.cat(sim[m][task]).mean().item()

        sim_table = pd.DataFrame(sim)
        sim_table.to_csv(os.path.join(self.args['log_path'], f'validation_sim.csv'))

        for task in self.eval_args['modality_modes']:
            task_name = serialize_task(task)
            self.writer.add_scalars(f'validation/{task_name}', {m: sim[m][task_name] for m in task['targets']}, current_step)

    def train_model(self, inputs, targets, reflector_inputs=None, enable_decode=False, current_step=0, pbar=None):
        self.ds_engine.module.train()

        loss, mle_acc, mse_loss = self.ds_engine(inputs, targets, 
                                                 reflector_inputs=reflector_inputs,
                                                 enable_decode=enable_decode)

        input_modality = [m for m in inputs if m != 'text']
        target_modality = list(targets.keys())
        if len(input_modality) > 1 or len(target_modality) > 1:
            task_type = 'multimodal'
        elif input_modality == target_modality:
            task_type = 'unimodal'
        else:
            task_type = 'crossmodal'

        self.writer.add_scalar(f'loss/all', loss.to(torch.float), current_step)
        self.writer.add_scalar(f'mse_loss/all', mse_loss.to(torch.float), current_step)
        self.writer.add_scalar(f'mle_acc/all', mle_acc, current_step)
        if enable_decode:
            self.writer.add_scalar(f'loss/decode_{task_type}', loss.to(torch.float), current_step)
            self.writer.add_scalar(f'mse_loss/decode_{task_type}', mse_loss.to(torch.float), current_step)
            self.writer.add_scalar(f'mle_acc/decode_{task_type}', mle_acc, current_step)
        else:
            self.writer.add_scalar(f'loss/{task_type}', loss.to(torch.float), current_step)
            self.writer.add_scalar(f'mse_loss/{task_type}', mse_loss.to(torch.float), current_step)
            self.writer.add_scalar(f'mle_acc/{task_type}', mle_acc, current_step)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}; mse_loss: {round(mse_loss.item(), 4)} ')
        pbar.update(1)
        if self.eval_args['validation_interval'] > 0 and current_step % self.eval_args['validation_interval'] == 0:
            torch.distributed.barrier()
            if self.args['local_rank'] == 0:
                self.eval(current_step)
        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        """
            this function also save the trainable parameters and specific name parameters
        """
        step_path = os.path.join(path, f'checkpoint_{current_step:07d}')
        os.makedirs(step_path)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.ds_engine.module.named_parameters()
        }
        state_dict = self.ds_engine.module.state_dict()
        # save llama
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad and 'reflector' not in k:
                checkpoint[k] = v
            if k in self.model.lora_params:
                checkpoint[k] = v
            if 'gen_text_hidden_fc' in k:
                checkpoint[k] = v
            if 'llama_proj' in k:
                checkpoint[k] = v
        torch.save(checkpoint, f'{step_path}/pytorch_model.pt')
        # save reflector
        checkpoint = OrderedDict()
        if self.args.get('enable_reflector') and not self.args['freeze_reflector']:
            for k, v in self.model.reflector.named_parameters():
                checkpoint[k] = v
            torch.save(checkpoint, f'{step_path}/reflector.pt')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(step_path)
        # save configuration
        self.model.llama_model.config.save_pretrained(step_path)
        print(f'[!] save model into {step_path}')

    def print_model_parameters(self, use_4bit=False):
        """
            Prints the number of trainable parameters in the model.
            """
        trainable_params = 0
        all_param = 0
        lora = 0
        image = 0
        linear = 0
        llama = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'gen_text_hidden_fc' in name:
                image += num_params
            elif 'llama_proj' in name:
                linear += num_params
            elif 'llama_model' in name:
                llama += num_params
            else:
                pass

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(f'lora params: {lora:,d} || image params: {image:,d}')
        print(f'linear params: {linear:,d} || llama params: {llama:,d}')

    def load_parameters(self, path):
        if os.path.exists(os.path.join(path, 'pytorch_model.pt')):
            print('loading parameters from {}'.format(path))
            delta_ckpt = torch.load(f'{path}/pytorch_model.pt', map_location=torch.device('cuda'))
            checkpoint = OrderedDict()
            skip_gen_text_hidden_fc = False
            if (self.model.state_dict()['gen_text_hidden_fc.query_embs'].shape !=
                delta_ckpt['gen_text_hidden_fc.query_embs'].shape):
                skip_gen_text_hidden_fc = True
                print(f'Skip loading gen_text_hidden_fc')
            for k, v in delta_ckpt.items():
                if 'gen_text_hidden_fc' in k and self.args.get('enable_decode') and skip_gen_text_hidden_fc:
                    continue
                if 'llama_model.model.embed_tokens.weight' in k:
                    checkpoint['llama_model.base_model.model.model.embed_tokens.weight'] = v
                elif 'llama_model.lm_head.weight' in k:
                    checkpoint['llama_model.base_model.model.lm_head.weight'] = v
                else:
                    checkpoint[k] = v
            self.model.load_state_dict(checkpoint, strict=False)
        if self.args.get('enable_reflector') and os.path.exists(os.path.join(path, 'reflector.pt')):
            print('loading reflector parameters from {}'.format(path))
            checkpoint = torch.load(f'{path}/reflector.pt', map_location=torch.device('cuda'))
            self.model.reflector.load_state_dict(checkpoint, strict=False)

