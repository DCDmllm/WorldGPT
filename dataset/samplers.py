# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""batch samplers that work with either random or sequential data samplers"""
import math
import os
import sys
import random

import torch
from torch.utils import data
import numpy as np


class RandomSampler(data.sampler.Sampler):
    r"""
    Based off of pytorch RandomSampler and DistributedSampler. Essentially a RandomSampler,
    but this class lets the user set an epoch like DistributedSampler
    Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.epoch = -1

        if self._num_samples is not None and replacement is False:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))
        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        g = torch.Generator()
        if self.epoch >= 0:
            g.manual_seed(self.epoch)
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=g).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64,
                                     generator=g).tolist()
        else:
            yield from torch.randperm(n, generator=self.generator).tolist()

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedSequentialSampler(data.sampler.Sampler):
    def __init__(self, num_samples, train_iters, batch_size, rank=-1, world_size=2):
        super().__init__(num_samples)
        if rank == -1:
            rank = 0
            world_size = 1
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.start_iter = 0
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.batch_bias = [i * (num_samples // batch_size) for i in range(batch_size)]

    def __iter__(self):
        for idx in range(self.start_iter, self.train_iters * 10):
            batch = [(idx + bias) % self.num_samples for bias in self.batch_bias]
            tbatch = self._batch(batch)
            yield tbatch

    def __len__(self):
        return self.train_iters

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank*self.batch_size//self.world_size
        end = (self.rank+1)*self.batch_size//self.world_size
        return batch[start:end]


class DistributedBatchSampler(data.sampler.BatchSampler):
    """
    similar to normal implementation of distributed sampler, except implementation is at the
    batch sampler level, instead of just the sampler level. This allows wrapping of arbitrary
    data samplers (sequential, random, WeightedRandomSampler, etc.) with this batch sampler.
    """
    def __init__(self, sampler, batch_size, drop_last, rank=-1, world_size=2, wrap_last=False, gradient_accumulation_steps=None):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        if rank == -1:
            assert False, 'should not be here'
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.effective_batch_size = batch_size if gradient_accumulation_steps is None else batch_size * gradient_accumulation_steps

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter * self.effective_batch_size:
                    yield tbatch
                    self.start_iter = 0
                i += len(batch)
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around%self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        start = self.rank*self.batch_size//self.world_size
        end = (self.rank+1)*self.batch_size//self.world_size
        return batch[start:end]


class DistributedWeightedMultiDatasetBatchSampler(data.sampler.BatchSampler):
    """
    This is a modality-blended batch sampler which allows to sample a batch data from different dataset alternatively.
    """
    def __init__(self, batch_size, dataset, weights, modality_modes, drop_last,
                 rank=-1, world_size=2, wrap_last=False, gradient_accumulation_steps=None):
        if rank == -1:
            assert False, 'should not be here'
        self.rank = rank
        self.world_size = world_size
        self.wrap_last = wrap_last
        self.drop_last = drop_last
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dataset = dataset
        self.weights = weights
        self.modality_modes = modality_modes
        self.modality_weights = [mode['weight'] for mode in modality_modes]
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([_cur_dataset.__len__() for _cur_dataset in dataset.datasets])

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = torch.utils.data.RandomSampler(cur_dataset)
            batch_sampler = DistributedBatchSampler(sampler, self.batch_size, self.drop_last, self.rank,
                                                    self.world_size, self.wrap_last, self.gradient_accumulation_steps)
            samplers_list.append(batch_sampler)
            cur_sampler_iterator = batch_sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        step = self.batch_size
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        from .state_dataset import StateDataset, CaptionDataset
        random_tasks = random.choices(self.modality_modes, weights=self.modality_weights, k=epoch_samples//step)
        for task in random_tasks:
            required_modality = list(set(task['inputs'] + task['targets']))
            available_datasets, dataset_weights = [], []
            for i in range(self.number_of_datasets):
                available = True
                if len(task['inputs']) == 0: # image-caption task
                    if not isinstance(self.dataset.datasets[i], CaptionDataset):
                        available = False
                else: # state-tranfer task
                    if not isinstance(self.dataset.datasets[i], StateDataset):
                        available = False
                for m in required_modality:
                    if m not in self.dataset.datasets[i].modality:
                        available = False
                        break
                if available:
                    available_datasets.append(i)
                    dataset_weights.append(self.weights[i])
            chosen_dataset = random.choices(available_datasets, weights=dataset_weights, k=1)[0]
            cur_batch_sampler = sampler_iterators[chosen_dataset]
            try:
                cur_sample_org = cur_batch_sampler.__next__()
                cur_samples = [(chosen_dataset, x, task) for x in cur_sample_org]
                yield cur_samples
            except StopIteration:
                # got to the end of iterator - restart the iterator and continue to get samples
                # until reaching "epoch_samples"
                sampler_iterators[chosen_dataset] = samplers_list[chosen_dataset].__iter__()
                cur_batch_sampler = sampler_iterators[chosen_dataset]
                cur_sample_org = cur_batch_sampler.__next__()
                cur_samples = [(chosen_dataset, x, task) for x in cur_sample_org]
                yield cur_samples

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size