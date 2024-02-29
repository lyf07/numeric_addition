import torch
import torch.nn
from torch.utils.data import Dataset
import numpy as np
import random

class AdditionDataSet(Dataset):
    def __init__(self, ndigits, split):
        '''
        @params: ndigits, eg: when ndigits==3, it means we have 3 digits + 3 digits = 3 / 4 digits(123 + 123 = 246, 999 + 999 = 1998)
        @params: split: train / test
        train : 80%
        test : 20%
        '''
        # super().__init__()
        self.ndigits = ndigits
        self.split = split
        self.vocab_size = 10
        self.block_size = 3 * ndigits
        # first we will generate the two source numbers
        all_conditions = (10 ** ndigits) ** 2
        # union generation
        # num = random.randint()
        generator = np.random.RandomState(1337)         # 1337 can be any arbitrary number
        perm = generator.permutation(all_conditions)
        border = int(all_conditions * 0.8)
        self.data = perm[:border] if self.split == 'train' else perm[border:]

    def __len__(self):
        return self.data.size

    def __getitem__(self, index):
        num = self.data[index]
        # split the first number and the second number
        base = 10 ** self.ndigits
        pre = num // base
        nxt = num % base
        result = pre + nxt
        render = f'%0{self.ndigits}d%0{self.ndigits}d%0{self.ndigits + 1}d' % (pre, nxt, result)
        data = [int(c) for c in render]
        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)
        # x = torch.tensor(data[:2 * self.ndigits], dtype=torch.long)
        # y = torch.tensor(data[2 * self.ndigits + 1:], dtype=torch.long)
        y[:self.ndigits * 2 - 1] = -100
        # y = torch.tensor(list(reversed(y)), dtype=torch.long)
        return x, y