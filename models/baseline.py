import os
import sys
import random
import json
import tqdm
import torch
import pyproj
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from planet_loader import PlanetDataset, collate_fn



class Module(nn.Module):
    def __init__(self, args, saved_model=None):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        if saved_model:
            self.args = saved_model['args']

        ### Layers Here ###
        self.linear_1 = nn.Linear(args.planet, 8)
        self.linear_2 = nn.Linear(8, 16)
        self.linear_3 = nn.Linear(16, args.planet * 3)

        if saved_model:
            self.load_state_dict(saved_model['model'], strict=False)

        ### End Layers ###

        self.to(self.device)

    def forward(self, t):
        ### Models Here ###
        args = self.args
        B = t.shape[0]
        t = t.repeat(1, args.planet).reshape(B, args.planet)
        t = t / 1E10

        positions = self.linear_1(t)
        positions = F.relu(positions)
        positions = self.linear_2(positions)
        positions = F.relu(positions)
        positions = self.linear_3(positions)
        positions = positions.reshape(B, args.planet, 3) * 1E10
        alt, az = self.convert_coordinates(positions[:,:,0],positions[:,:,1],positions[:,:,2])

        positions = torch.stack([az, alt], dim=-1)
        positions = positions.reshape(B, args.planet * 2)
        return positions
        ### End Model ###

    def gps_to_ecef_custom(self, lat, lon, alt):
        rad_lat = lat * torch.tensor(np.pi / 180.0).to(self.device)
        rad_lon = lon * torch.tensor(np.pi / 180.0).to(self.device)

        a = torch.tensor(6378137, dtype=torch.float).to(self.device)
        finv = 298.257223563
        f = 1 / finv
        e2 = 1 - (1 - f) * (1 - f)
        e2 =  torch.tensor(e2, dtype=torch.float).to(self.device)
        v = torch.div(a, torch.sqrt(1 - e2 * torch.sin(rad_lat) * torch.sin(rad_lat)))

        x = (v + alt) * torch.cos(rad_lat) * torch.cos(rad_lon)
        y = (v + alt) * torch.cos(rad_lat) * torch.sin(rad_lon)
        z = (v * (1 - e2) + alt) * torch.sin(rad_lat)

        return x, y, z



    def ecef2lla(self, x, y, z) :
        B = x.shape[0]
        a = torch.tensor(6378137, dtype=torch.float).to(self.device)
        f = torch.tensor(0.0034, dtype=torch.float).to(self.device)
        b = torch.tensor(6.3568e6, dtype=torch.float).to(self.device)
        e = torch.sqrt(torch.div(torch.pow(a, 2) - torch.pow(b, 2) , torch.pow(a, 2)))
        e2 = torch.sqrt(torch.div(torch.pow(a, 2) - torch.pow(b, 2) , torch.pow(b, 2)))

        lla = torch.zeros(3, B, self.args.planet).to(self.device)

        p = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2))

        theta = torch.arctan(torch.div((z * a), (p * b)))

        lon = torch.arctan(torch.div(y, x))

        first = z + torch.pow(e2, 2) * b * torch.pow(torch.sin(theta), 3)
        second = p - torch.pow(e, 2) * a * torch.pow(torch.cos(theta), 3)
        lat = torch.arctan(torch.div(first, second))
        N = torch.div(a, (torch.sqrt(1 - (torch.pow(e, 2) * torch.pow(torch.sin(lat), 2)))))

        m = torch.div(p, torch.cos(lat))
        height = m - N

        lon = torch.div(lon * 180, torch.tensor(np.pi)).to(self.device)
        lat = torch.div(lat * 180, torch.tensor(np.pi)).to(self.device)
        lla[0] = lat
        lla[1] = lon
        lla[2] = height

        return lla


    def bearing(self, lat1, lon1, lat2, lon2):
        y = torch.sin(lon2-lon1) * torch.cos(lat2)
        x = torch.cos(lat1)*torch.sin(lat2) - torch.sin(lat1)*torch.cos(lat2)*torch.cos(lon2-lon1)
        theta = torch.atan2(y, x)
        brng = torch.fmod(theta*180/np.pi + 360, 360)
        return brng

    def angle_between(self, v1, v2):
        v1_u = torch.div(v1 , torch.norm(v1, dim=1).unsqueeze(1).repeat(1, 3))
        v1_u = v1_u.unsqueeze(1)
        # print(v1_u.shape)
        v2_u = torch.div(v2 , torch.norm(v2, dim=2).unsqueeze(2).repeat(1, 1, 3))
        return torch.arccos(torch.clip(torch.bmm(v1_u, torch.transpose(v2_u, 1, 2)), -1.0, 1.0)).squeeze()

    def convert_coordinates(self, x, y, z):
        x_pos, y_pos, z_pos = self.gps_to_ecef_custom(self.args.latitude, self.args.longtitude, self.args.alt)
        B = x.shape[0]
        b_x = x_pos.expand(B,self.args.planet).to(self.device)
        b_y = y_pos.expand(B,self.args.planet).to(self.device)
        b_z = z_pos.expand(B,self.args.planet).to(self.device)

        a_x = x_pos.expand(B).to(self.device)
        a_y = y_pos.expand(B).to(self.device)
        a_z = z_pos.expand(B).to(self.device)

        a = torch.stack([a_x, a_y, a_z], dim=-1)
        b = torch.stack([x-b_x, y-b_y, z-b_z], dim=-1)
        altitude = self.angle_between(a, b)

        lon2, lat2, _ = self.ecef2lla(x, y, z)
        lat1 = torch.tensor(self.args.latitude).repeat(B, self.args.planet)
        lon1 = torch.tensor(self.args.longtitude).repeat(B, self.args.planet)
        azimuth = self.bearing(lat1, lon1, lat2, lon2)


        return altitude, azimuth

    def run_train(self, data):
        ### Setup ###
        args = self.args
        self.writer = SummaryWriter(args.writer) ## Our summary writer, which plots loss over time
        self.fsave = os.path.join(args.dout) ## The path to the best version of our model

        with open(data, 'r') as file:
            dataset = json.load(file)
        dataset = PlanetDataset(dataset, self.args)
        train_size = int(len(dataset) * 0.8)
        valid_size = len(dataset) - val1
        train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(list(self.parameters()), lr=args.lr)

        best_loss = 1e10 ## This will hold the best validation loss so we don't overfit to training data

        ### Training Loop ###
        batch_idx = 0
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            train_description = "Epoch " + str(epoch) + ", train"
            valid_description = "Epoch " + str(epoch) + ", valid"

            train_loss = torch.tensor(0, dtype=torch.float)
            train_size = torch.tensor(0, dtype=torch.float)

            self.train()



            for batch in tqdm.tqdm(train_dataloader, desc=train_description):
                batch_idx+=1
                optimizer.zero_grad() ## You don't want to accidentally have leftover gradients

                ### Compute loss and update variables ###
                loss = self.compute_loss(batch)

                train_size += batch['time'].shape[0] ## Used to normalize loss when plotting
                train_loss += loss

                self.writer.add_scalar("Loss/BatchTrain", (loss/batch['time'].shape[0]).item(), batch_idx)

                ### Backpropogate ###
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            ### Run validation loop ###
            valid_loss, valid_size = self.run_eval(valid_dataloader, valid_description)

            ### Write to SummaryWriter ###
            self.writer.add_scalar("Loss/Train", (train_loss/train_size).item(), epoch)
            self.writer.add_scalar("Loss/Valid", (valid_loss/valid_size).item(), epoch)
            self.writer.flush()

            ### Save model if it is better that the previous ###

            if valid_loss < best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format((valid_loss/valid_size).item(), self.fsave))
                torch.save({
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args
                }, self.fsave)
                best_loss = valid_loss

        self.writer.close()

    def evaluate(self, data):
        self.eval()
        with open(data, 'r') as file:
            dataset = json.load(file)
        dataset = PlanetDataset(dataset, self.args)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
        description = "Evaluating..."
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=description):
                size += batch['time'].shape[0] ## Used to normalize loss when plotting
                loss += self.compute_loss(batch)
        return (loss / size).item()


    def compute_loss(self, batch):
        batch_size = batch['time'].shape[0]

        ### Move data to GPU if available ###
        times = batch['time'].to(self.device)
        true_positions = batch['pos'].to(self.device)

        ### Run forward ###
        pred_positions = self.forward(times)

        ### Compute loss on results ###
        loss = F.mse_loss(pred_positions, true_positions, reduction='sum')
        return loss
