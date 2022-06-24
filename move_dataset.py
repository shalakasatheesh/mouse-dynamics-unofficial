import itertools
import pickle
from typing import Any
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import re
import numpy as np
import torch.nn.functional as F


class MoveData:

    def __init__(self, time, pos):
        self.time = time
        self.pos = pos


class MoveBlockData:

    def __init__(self, token, move_data):
        self.token = token
        self.move_data = move_data


class MoveDataset(Dataset):

    def __init__(self, csv_folder=None, pickle_file=None, transform=None):
        if pickle_file is None:
            self.csv_folder = csv_folder
            self.files = os.listdir(csv_folder)
            self.block_data = []
            for i, file in enumerate(self.files):
                percentage = float(i) / float(len(self.files)) * 100.0
                print(str(percentage) + '%', 'Processing file', file, '...')
                self.load_csv_file(file)
        else:
            self.block_data = pickle.load(open(pickle_file, 'rb'))

        self.user_ids = {}
        self.assign_user_ids()
        self.transform = transform

    def train_data(self):
        X = np.empty(shape=(len(self), 128, 2))
        labels = np.zeros(shape=(len(self), self.unique_user_count()))
        for idx, (x, y) in enumerate(self):
            X[idx] = x.numpy()
            labels[idx, y] = 1.0
        return X, labels

    def assign_user_ids(self):
        counter = 0
        for t in set([d.token for d in self.block_data]):
            self.user_ids[t] = counter
            counter += 1

    def unique_user_count(self):
        return len(self.user_ids)

    def __getitem__(self, index: Any):
        move_block = self.block_data[index]
        t = torch.zeros((128, 2))

        for i, d in enumerate(move_block.move_data):
            t[i, 0] = d[0]
            t[i, 1] = d[1]

        if self.transform:
            t = self.transform(t)

        return t, self.user_ids[move_block.token]

    def __len__(self):
        return len(self.block_data)

    def generate_dx_dy(self, token, data):
        new_data = [(0, 0)]
        for idx, value in enumerate(data[1:], start=0):
            dx = value.pos[0] - data[idx].pos[0]
            dy = value.pos[1] - data[idx].pos[1]
            new_data.append((dx, dy))
        return MoveBlockData(token, new_data)

    def process_block_data(self, token, data):
        block_data = []
        if len(data) == 0:
            return block_data

        current_idx = 0

        while current_idx < len(data):
            starting_time = data[current_idx].time
            remaining = [data[current_idx]]
            remaining.extend(
                itertools.takewhile(lambda d: (d.time - starting_time).total_seconds() <= 15,
                                    data[current_idx + 1:])
            )

            length = 0
            if len(remaining) > 128:
                remaining = remaining[:128]
                length = 128
            elif len(remaining) < 128:
                length = len(remaining)
                for _ in range(128 - len(remaining)):
                    remaining.append(remaining[-1])

            if length == 128:
                block_data.append(self.generate_dx_dy(token, remaining))
            current_idx += length + 1

        return block_data

    def load_csv_file(self, file):

        df = pd.read_csv(os.path.join(self.csv_folder, file), sep=';')
        df['time'] = pd.to_datetime(df['time'])
        token = df['token'][0]

        data = []
        for idx, row in df.iterrows():
            t = re.split('[]\[,]', row['global_position'])
            data.append(MoveData(row['time'], (int(t[1]), int(t[2]))))

        b_data = self.process_block_data(token, data)
        if len(b_data) > 0:
            self.block_data.extend(b_data)

    def save_as_pickle(self, pickle_file):
        pickle.dump(self.block_data, open(pickle_file, "wb"))


class Normalize(object):

    def __call__(self, sample):
        dx_col = sample[:, 0]
        dy_col = sample[:, 1]

        # Normalize the dx values
        std = torch.std(dx_col)
        if std == 0:
            std = 0.0001
        mean = torch.mean(dx_col)
        sample[:, 0] = (dx_col - mean) / std

        # Normalize the dy values
        std = torch.std(dy_col)
        if std == 0:
            std = 0.0001
        mean = torch.mean(dy_col)
        sample[:, 1] = (dy_col - mean) / std

        return sample
        # return F.normalize(sample)
