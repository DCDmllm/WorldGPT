import random

import torch
from torch.utils.data import DataLoader
from .concat_dataset import ConcatDataset, DecoderConcatDataset
from .samplers import DistributedBatchSampler, DistributedWeightedMultiDatasetBatchSampler
from .state_dataset import StateDataset, CaptionDataset, collate_embeds, collate_path
from .decoder_state_dataset import DecoderStateDataset, collate_decoder


def load_dataset(args, dataset_list):
    state_datasets = []
    for metadata in dataset_list:
        if metadata.get('image_caption'):
            state_datasets.append(CaptionDataset(metadata))
        elif args['precomputed_languagebind']:
            state_datasets.append(StateDataset(metadata, return_path=False))
        else:
            state_datasets.append(StateDataset(metadata, return_embeds=False))

    if args['mode'] != 'train':
        concat_data = torch.utils.data.ConcatDataset(state_datasets)
        batch_size = args['dschf'].config['train_micro_batch_size_per_gpu']
        iter_ = DataLoader(
            concat_data,
            num_workers=1,
            batch_size=batch_size,
            drop_last=False,
            collate_fn=collate_embeds if args.get('precomputed_languagebind') else collate_path,
            pin_memory=True
        )
        return concat_data, iter_
        
    concat_data = ConcatDataset(state_datasets)
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    batch_sampler = DistributedWeightedMultiDatasetBatchSampler(dataset=concat_data,
                                                                batch_size=batch_size,
                                                                weights=[i['weight'] for i in dataset_list],
                                                                modality_modes=args['modality_modes'],
                                                                drop_last=True,
                                                                rank=args['local_rank'],
                                                                world_size=args['world_size'])
    iter_ = DataLoader(
        concat_data,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=collate_embeds if args.get('precomputed_languagebind') else collate_path,
        pin_memory=True
    )
    return concat_data, iter_


def load_decoder_dataset(args, dataset_list, train_modality,
                         state0_transforms=None, state1_transforms=None,
                         rank=0, world_size=1):
    state_datasets = []
    for metadata in dataset_list:
        state_datasets.append(DecoderStateDataset(metadata, train_modality,
                                                  state0_transforms=state0_transforms,
                                                  state1_transforms=state1_transforms))
    concat_data = DecoderConcatDataset(state_datasets)
    batch_sampler = DistributedWeightedMultiDatasetBatchSampler(dataset=concat_data,
                                                                batch_size=args['train_batch_size'],
                                                                weights=[i['weight'] for i in dataset_list],
                                                                modality_modes=args['train_modality_modes'],
                                                                drop_last=True,
                                                                rank=rank, world_size=world_size)
    iter_ = DataLoader(
        concat_data,
        batch_sampler=batch_sampler,
        num_workers=1,
        collate_fn=collate_decoder,
        pin_memory=True
    )
    return concat_data, iter_