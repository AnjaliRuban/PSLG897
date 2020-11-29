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

        self.planet_sidereals = {
            "mercury": 87,
            "mars": 686,
            "venus": 224,
            "jupiter": 4332,
            "saturn": 10765
        }


    def __getitem__(self, idx):
        item = self.data[idx]
        item = self.featurize(item)
        return item

    def __len__(self):
        return len(self.data)

    def featurize(self, item):
        time = item['time']
        # print(time)
        # time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M").timestamp()  ## In POSIX time, might pose a problem if we go too far back w/ sig figs?
        real_time = 365.25 * int(time[:4]) + 30.44 * int(time[5:7]) + int(time[8:10])
        # real_time = (365.25) + 30.44 * int(time[5:7]) + int(time[8:10])
        positions = []
        planet_times = []
        for planet in sorted(item['planet_data'].keys()):
            az = item['planet_data'][planet]['az']
            alt = item['planet_data'][planet]['alt']
            positions.append([az, alt])
            planet_times.append(real_time % self.planet_sidereals[planet])
        real_time = torch.tensor(real_time)
        planet_times = torch.tensor(planet_times)
        positions = torch.tensor(positions)

        return {
            'time': real_time,
            'planet_times': planet_times,
            'pos': positions
        }

def collate_fn(batch):
    batch_size = len(batch)

    times = []
    planet_times = []
    positions = []
    for feat in batch:
        times.append(feat['time'])
        planet_times.append(feat['planet_times'])
        positions.append(feat['pos'])

    times = torch.stack(times)
    positions = torch.stack(positions)
    planet_times = torch.stack(planet_times)

    return {
        'time': times,
        'planet_times': planet_times,
        'pos': positions
    }
