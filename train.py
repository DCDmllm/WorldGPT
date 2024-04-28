import os
import time
import argparse
import random
import logging
from tqdm import tqdm

import torch
import numpy as np
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from dataset import *
from model.agent import DeepSpeedAgent
from model.worldgpt import WorldGPTModel
from model.preprocessor import PreProcessor
from config import load_config


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--log_path', type=str, default=None)

    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--cfg_path', type=str, required=True)
    parser.add_argument('--eval_only', default=False, action='store_true')
    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    if os.path.exists(path):
        pass
    else:  # recursively construct directory
        os.makedirs(path, exist_ok=True)


def main(**args):
    config_env(args)
    print(args)
    args['ds_config_path'] = f'config/ds_base.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    if args['dschf']['fp16']['enabled']:
        args['dtype'] = torch.float16
    elif args['dschf']['bf16']['enabled']:
        args['dtype'] = torch.bfloat16
    else:
        args['dtype'] = torch.float

    if args.get('save_path') is None:
        args['save_path'] = os.path.join('log', os.path.basename(args['cfg_path']).rpartition('.')[0])
    if not args['eval_only']:
        build_directory(args['save_path'])
    if args['log_path'] is None:
        args['log_path'] = os.path.join(args['save_path'], 'log')
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )

    if not args['eval_only']:
        train_data, train_iter = load_dataset(args, args['dataset_list'])
        # train_num = max([_cur_dataset.__len__() for _cur_dataset in train_data.datasets]) * len(train_data.datasets)
        step_per_epoch = min(
            max([_cur_dataset.__len__() for _cur_dataset in train_data.datasets]) * len(train_data.datasets),
            args['samples_per_epoch']
        ) // args['world_size'] // args['dschf'].config['train_micro_batch_size_per_gpu']
        length = args['epochs'] * step_per_epoch
        total_steps = args['epochs'] * step_per_epoch * args['world_size']
        args['total_steps'] = total_steps
    else:
        args['total_steps'] = 0
        args['warmup_rate'] = 0

    device = torch.cuda.current_device()
    model = WorldGPTModel(**args)
    agent = DeepSpeedAgent(model, args)
    if not args['precomputed_languagebind'] or args.get('enable_reflector'):
        preprocessor = PreProcessor(args)
    torch.distributed.barrier()

    if args['eval_only']:
        agent.eval(0)
        return

    # begin to train
    pbar = tqdm(total=length)  # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        # for train_iter in train_iter_list:
        for step, (inputs, targets) in enumerate(train_iter):
            if step >= step_per_epoch:
                break
            reflector_inputs = None
            if not args['precomputed_languagebind']:
                inputs, _ = preprocessor(inputs)
                targets, _ = preprocessor(targets)
            if args.get('enable_reflector'):
                _, action_embeds = preprocessor({'text': inputs['text']})
                state0 = torch.stack([inputs[m] for m in inputs if m != 'text'], dim=1).mean(dim=1).to(device)
                state1 = torch.stack([targets[m] for m in targets], dim=1).mean(dim=1).to(device)
                reflector_inputs = {
                    'state0': state0,
                    'action': action_embeds,
                    'context': torch.stack([state0, action_embeds, state1], dim=1)
                }
            agent.train_model(
                inputs, targets, reflector_inputs,
                current_step=current_step,
                pbar=pbar
            )
            if current_step % args['save_interval'] == 0:
                torch.distributed.barrier()
                if args['local_rank'] == 0:
                    agent.save_model(args['save_path'], current_step)
            current_step += 1
    # save at the end of the training
    torch.distributed.barrier()
    if args['local_rank'] == 0:
        agent.save_model(args['save_path'], current_step)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
