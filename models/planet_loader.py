import os
import sys
import json
import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PlanetDataset(Dataset):
    def __init__(self, data, args):
        self.args = args
        self.data = data


    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.featurize(item)
        return item

    def __len__(self):
        return len(self.data)

    def featurize(self, item):
        time = item['time']
        time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M").timestamp()  ## In POSIX time, might pose a problem if we go too far back w/ sig figs?
        positions = []
        for planet in item['planet_data'].keys():
            az = item['planet_data'][planet]['az']
            alt = item['planet_data'][planet]['alt']
            positions.append([az, alt])

        time = torch.tensor(time)
        positions = torch.tensor(positions)

        return {
            'time': time,
            'pos': positions
        }

def collate_fn(batch):
    batch_size = len(batch)

    times = []
    positions = []
    for feat in batch:
        times.append(feat['time'])
        positions.append(feat['pos'].reshape(-1))

    times = torch.stack(times)
    positions = torch.stack(positions)

    return {
        'time': times,
        'pos': positions
    }
