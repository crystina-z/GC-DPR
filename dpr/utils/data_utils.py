#!/usr/bin/env python3
# Copyright GC-DPR authors.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""

import json
import logging
import math
import os
import pickle
import random
from typing import List, Dict, Iterator, Callable, Union

from torch import Tensor as T
from torch.utils.data import IterableDataset

logger = logging.getLogger()


def get_data_size(data):
    assert isinstance(data, list)
    assert all(isinstance(d, dict) for d in data)
    if all(set(d.keys()) == {"meta", "data"} for d in data):
        # assume each data entry has format {"meta": meta, "data": data}
        return sum(len(sample["data"]) for sample in data)
    else:
        return len(data)


def remove_data_wo_pos_ctx(data):
    assert isinstance(data, list)
    # if all(isinstance(d, dict) for d in data) and set(data) == {"meta", "data"}:
    assert all(isinstance(d, dict) for d in data)
    if all(set(d.keys()) == {"meta", "data"} for d in data):
        # assume each data entry has format {"meta": meta, "data": data}
        # return sum(len(sample["data"]) for sample in data)
        # return [r for r in data if len(r['data']['positive_ctxs']) > 0]
        cleaned = [{
            "meta": meta_data["meta"],
            "data": [r for r in meta_data["data"] if len(r['positive_ctxs']) > 0]
        } for meta_data in data]
    else:
        cleaned = [r for r in data if len(r['positive_ctxs']) > 0]

    data_size = get_data_size(cleaned)
    logger.info('Total cleaned data size: {}'.format(data_size))
    return cleaned


def load_data_from_json(json_f, aggregated, upsample_factor: int = 1) -> List:
    data = json.load(json_f)
    if isinstance(data, list):
        data = data * upsample_factor
        aggregated.extend(data)
    elif isinstance(data, dict):
        assert set(data) == {"meta", "data"}, "The loaded dict should contain both 'meta' and 'data' field"
        meta = data["meta"]
        data = data["data"]
        print("meta", type(meta), "data: ", type(data), flush=True)
        assert isinstance(meta, dict) and isinstance(data, list)
        data = data * upsample_factor
        aggregated.append({"meta": meta, "data": data})
    else:
        raise ValueError(
            "Unexpected data type, should be either a list of data or a dict containing 'meta' and 'data' fields"
        )
    data_size = get_data_size(aggregated)
    logger.info('Aggregated data size: {}'.format(data_size))
    return aggregated


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info('Reading file %s', path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info('Aggregated data size: {}'.format(len(results)))
    logger.info('Total data size: {}'.format(len(results)))
    return results


def read_data_from_json_files(paths: List[str], upsample_rates: List = None) -> Union[List, Dict]:
    results = []
    if upsample_rates is None:
        upsample_rates = [1] * len(paths)

    assert len(upsample_rates) == len(paths), 'up-sample rates parameter doesn\'t match input files amount'

    for i, path in enumerate(paths):
        with open(path, 'r', encoding="utf-8") as f:
            logger.info('Reading file %s' % path)
            assert path.endswith(".json")
            upsample_factor = int(upsample_rates[i])
            load_data_from_json(
                json_f=f,
                aggregated=results,
                upsample_factor=upsample_factor,
            )  # update results internally
            # data = json.load(f)
            # data = data * upsample_factor
            # results.extend(data)
            # logger.info('Aggregated data size: {}'.format(len(results)))
    return results


class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    @staticmethod
    def group_data(data):
        """ we only consider lang in the group key now """
        assert isinstance(data, list)
        assert all(isinstance(d, dict) for d in data)
        if all(set(d.keys()) == {"meta", "data"} for d in data):
            assert all("lang" in d["meta"] for d in data)
            data = {d["meta"]["lang"]: d["data"] for d in data}
        else:
            # if each entry in the list is of key {"question": ..., {"answers": ...}}, then it does not have meta info
            data = {"combined": data}
        return data

    def __init__(self, data: list, shard_id: int = 0, num_shards: int = 1, batch_size: int = 1, shuffle=True,
                 shuffle_seed: int = 0, offset: int = 0,
                 strict_batch_size: bool = False
                 ):

        data = self.group_data(data)
        self.data = data
        # total_size = len(data)

        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)
        _, samples_per_shard = self.init_data_per_shard()
        self.samples_per_shard = samples_per_shard
        # samples_per_shard = math.ceil(self.total_size / self.shards_num)
        # self.shard_start_idx = self.shard_id * samples_per_shard
        # self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, self.total_size)

        if strict_batch_size:
            self.max_iterations = math.ceil(samples_per_shard / batch_size)
        else:
            self.max_iterations = int(samples_per_shard / batch_size)

        logger.debug(
            # 'samples_per_shard=%d, shard_start_idx=%d, shard_end_idx=%d, max_iterations=%d', total_samples_per_shard,
            # self.shard_start_idx,
            # self.shard_end_idx,
            'samples_per_shard=%d, max_iterations=%d', samples_per_shard, self.max_iterations)

        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size

    @property
    def total_size(self):
        if not hasattr(self, "_total_size"):
            assert hasattr(self, "data")
            self._total_size = sum(len(data_sample) for data_sample in self.data.values())
        return self._total_size

    def init_data_per_shard(self):
        """
        Assume that this function is called after the data has been grouped
        return: total_data_size, total_data_size_per_shard
        """
        self.sample_size = {g_name: len(self.data[g_name]) for g_name in self.data}
        self.shard_samples_num = {g_name: math.ceil(self.sample_size[g_name] / self.shards_num) for g_name in self.data}
        self.shard_start_idx = {g_name: self.shard_id * self.shard_samples_num[g_name] for g_name in self.data}
        self.shard_end_idx = {g_name: min(
            self.shard_start_idx[g_name] + self.shard_samples_num[g_name],
            self.sample_size[g_name]
        ) for g_name in self.data}
        return map(lambda dct: sum(dct.values()), [self.sample_size, self.shard_samples_num])

    def get_batch(self, batch_idx, g_name):
        """return a batch of data from the same group"""
        start_idx = self.shard_start_idx[g_name]
        end_idx = self.shard_end_idx[g_name]
        shard_samples = self.data[g_name][start_idx:end_idx]
        items = shard_samples[batch_idx:batch_idx + self.batch_size]
        if self.strict_batch_size and len(items) < self.batch_size:
            logger.debug('Extending batch to max size')
            items.extend(shard_samples[0:self.batch_size - len(items)])
        return items

    def next_group_name(self):
        """Infinite loop that always yield next group to check"""
        while True:
            for g_name in self.data:
                yield g_name

    def shuffle_data(self, rnd):
        """shuffle data within each group"""
        for g_name in self.data:
            rnd.shuffle(self.data[g_name])

    def iterate_data(self, epoch: int = 0) -> Iterator[List]:
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            # epoch_rnd.shuffle(self.data)
            self.shuffle_data(epoch_rnd)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        max_iterations = self.max_iterations - self.iteration
        group_name_iter = self.next_group_name()

        for i in range(self.iteration * self.batch_size, self.samples_per_shard, self.batch_size):
            # for g_name in self.data:
            g_name = next(group_name_iter)
            items = self.get_batch(batch_idx=i, g_name=g_name)
            self.iteration += 1
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            g_name = next(group_name_iter)
            items = self.get_batch(batch_idx=0, g_name=g_name)
            yield items
            # batch = shard_samples[0:self.batch_size]
            # yield batch

        logger.debug('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for g_name in self.data:
            for sample in self.data[g_name]:
                visitor_func(sample)


class ShardedDataIterableDataset(ShardedDataIterator, IterableDataset):
    def __init__(self, *args, process_fn=None, **kwargs):
        ShardedDataIterator.__init__(self, *args, **kwargs)

        self.epoch = None
        self.curr_idx = None
        self.idx_gen = None
        self.shard_samples = None
        self._max_iterations = None
        self._ended = False
        self.process_fn = process_fn

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + self.epoch)
            # epoch_rnd.shuffle(self.data)
            self.shuffle_data(epoch_rnd)

        # if resuming iteration somewhere in the middle of epoch, one needs to adjust max_iterations
        self._max_iterations = self.max_iterations - self.iteration
        # self.shard_samples = self.data[self.shard_start_idx:self.shard_end_idx]
        # self.idx_gen = iter(range(self.iteration * self.batch_size, len(self.shard_samples), self.batch_size))
        self.idx_gen = iter(range(self.iteration * self.batch_size, self.samples_per_shard, self.batch_size))
        self.iteration = 0
        self._ended = False
        self.group_name_iter = self.next_group_name()
        return self

    def __next__(self):
        if not self._ended:
            try:
                i = next(self.idx_gen)
                g_name = next(self.group_name_iter)
                items = self.get_batch(batch_idx=i, g_name=g_name)
                # items = self.shard_samples[i:i + self.batch_size]
                # print("id: ", i, "end:", i + self.batch_size, "item sample:", len(items), items)
                print("id: ", i, "end:", i + self.batch_size, "item sample:", len(items))
                # seems to pass
                # for it in items:
                #     print("\t", it["question"])
                # print("=" * 100)
                if self.strict_batch_size:
                    assert len(items) == self.batch_size

                # if self.strict_batch_size and len(items) < self.batch_size:
                #     logger.debug('Extending batch to max size')
                #     items.extend(self.shard_samples[0:self.batch_size - len(items)])
                self.iteration += 1
                if self.process_fn:
                    random.seed(self.shuffle_seed + self.epoch + self.iteration)
                    return self.process_fn(items)
                return items
            except StopIteration:
                self._ended = True

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        if self.iteration < self._max_iterations:
            logger.debug('Fulfilling non complete shard='.format(self.shard_id))
            self.iteration += 1
            # batch = self.shard_samples[0:self.batch_size]
            g_name = next(self.group_name_iter)
            batch = self.get_batch(batch_idx=0, g_name=g_name)
            if self.process_fn:
                random.seed(self.shuffle_seed + self.epoch + self.iteration)
                return self.process_fn(batch)
            return batch

        logger.info('Finished iterating, iteration={}, shard={}'.format(self.iteration, self.shard_id))
        # reset the iteration status
        raise StopIteration


def normalize_question(question: str) -> str:
    if question[-1] == '?':
        question = question[:-1]
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError
