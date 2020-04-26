from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
import random
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class TripletSampler(Sampler):
    def __init__(self, data_source, pos, neg):
        self.data_source = data_source
        self.pos = pos
        self.neg = neg

    def __len__(self):
        return len(self.data_source) * 3

    def __iter__(self):
        indices = torch.randperm(len(self.data_source))
        ret = []
        for i in indices:
            if len(self.pos[i]) == 0 or len(self.neg[i]) == 0:
                continue
            ret.append(i)
            ret.append(random.choice(self.pos[i]))
            ret.append(random.choice(self.neg[i]))
        return iter(ret)
