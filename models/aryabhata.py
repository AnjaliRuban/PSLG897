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
        self.linear_theta1 = nn.Linear(1, 1)
        self.linear_radius1 = nn.Linear(1, 1)

        self.linear_manda = nn.Linear(1, 1)

        self.linear_phi2 = nn.Linear(1, args.planet)
        self.linear_theta2 = nn.Linear(1, args.planet)
        self.linear_radius2 = nn.Linear(1, args.planet)

        if saved_model:
            self.load_state_dict(saved_model['model'], strict=False)
        ### End Layers ###

        self.to(self.device)

    def forward(self, t):
        ### Models Here ###
        args = self.args
        B = t.shape[0]
        t = t.unsqueeze(1) / 1E14

        theta_1 = self.linear_theta1(t) # -> B x 1
        radius_1 = self.linear_radius1(t) # -> B x 1

        # print("weights", self.linear_theta1.weight)
        # print("bias", self.linear_theta1.bias)

        mean_sun_x = torch.sin(theta_1) * radius_1 # -> B x 1
        mean_sun_y = torch.cos(theta_1) * radius_1 # -> B x 1
        mean_sun_z = torch.zeros(B, dtype=torch.float).to(self.device).unsqueeze(1) # -> B x 1



        manda_x = mean_sun_x.expand(B, args.planet)
        manda_y = mean_sun_y.expand(B, args.planet)
        manda_z = self.linear_manda(mean_sun_z).expand(B, args.planet)

        phi_2 = self.linear_phi2(torch.ones(B, dtype=torch.float).to(self.device).unsqueeze(1))
        theta_2 = self.linear_theta2(t)
        radius_2 = self.linear_radius2(torch.ones(B, dtype=torch.float).to(self.device).unsqueeze(1))

        # phi_2 = self.linear_phi2(t)
        # theta_2 = self.linear_theta2(t)
        # radius_2 = self.linear_radius2(t)

        # print("phi2", phi_2)
        # print("theta2", theta_2)
        # print("phi2", phi_2)

        sighara_x = (radius_2 * torch.sin(phi_2) * torch.cos(theta_2)) + manda_x
        sighara_y = (radius_2 * torch.sin(phi_2) * torch.sin(theta_2)) + manda_y
        sighara_z = (radius_2 * torch.cos(phi_2)) + manda_z


        # print(sighara_z.shape)

        # print("manda_x", manda_x[0])
        # print("costheta", torch.cos(theta_2)[0])
        # print("sinphi", torch.sin(phi_2)[0])
        # print("radius", radius_2[0])

        alt, az = self.convert_coordinates(sighara_x * 1E14, sighara_y * 1E14, sighara_z * 1E14)
        positions = torch.stack([az, alt], dim=-1)
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
        v2_u = torch.div(v2 , torch.norm(v2, dim=2).unsqueeze(2).repeat(1, 1, 3))
        # print("v1_u", v1_u)
        # print("v2_u", v2_u)
        return torch.arccos(torch.clip(torch.bmm(v1_u, torch.transpose(v2_u, 1, 2)), -1.0, 1.0)).squeeze(1)

    def convert_coordinates(self, x, y, z):
        x_pos, y_pos, z_pos = self.gps_to_ecef_custom(self.args.latitude, self.args.longtitude, self.args.alt)
        x_pos = x_pos
        y_pos = y_pos
        z_pos = z_pos
        # print("Xpos", x_pos)
        # print("Ypos", y_pos)
        # print("Zpos", z_pos)
        B = x.shape[0]
        b_x = x_pos.expand(B,self.args.planet).to(self.device)
        b_y = y_pos.expand(B,self.args.planet).to(self.device)
        b_z = z_pos.expand(B,self.args.planet).to(self.device)

        # print("x", x[0])
        # print("y", y[0])
        # print("z", z[0])

        a_x = x_pos.expand(B).to(self.device)
        a_y = y_pos.expand(B).to(self.device)
        a_z = z_pos.expand(B).to(self.device)

        a = torch.stack([a_x, a_y, a_z], dim=-1)
        b = torch.stack([x-b_x, y-b_y, z-b_z], dim=-1)

        altitude = self.angle_between(a, b)

        # print("altitude", altitude[0])


        # lon2, lat2, _ = self.ecef2lla(x / 1E3, y / 1E3, z / 1E3)
        # print("lon2 after", lon2)

        lon2, lat2, _ = self.ecef2lla(x, y, z)
        # print("lon2 before", lon2)

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
        valid_size = len(dataset) - train_size
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

    def run_eval(self, valid_dataloader, valid_description):
        self.eval()
        loss = torch.tensor(0, dtype=torch.float)
        size = torch.tensor(0, dtype=torch.float)

        with torch.no_grad():
            for batch in tqdm.tqdm(valid_dataloader, desc=valid_description):
                size += batch['time'].shape[0] ## Used to normalize loss when plotting
                loss += self.compute_loss(batch)

        return loss, size

    def evaluate(self, data):
        self.eval()
        args = self.args
        with open(data, 'r') as file:
            dataset = json.load(file)
        dataset = PlanetDataset(dataset, self.args)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
        description = "Evaluating..."

        loss = [torch.tensor(0, dtype=torch.float) for i in range(self.args.planet)]
        size = torch.tensor(0, dtype=torch.float)

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=description):
                batch_size = batch['time'].shape[0]

                ### Move data to GPU if available ###
                times = batch['time'].to(self.device)
                true_position = batch['pos'].to(self.device)

                ### Run forward ###
                pred_position = self.forward(times)

                true = []
                pred = []
                true_az_sin = torch.sin(torch.deg2rad(true_position[:, :, 0]))
                true_az_cos = torch.cos(torch.deg2rad(true_position[:, :, 0]))
                true_alt_sin = torch.sin(torch.deg2rad(true_position[:, :, 1]))
                true_alt_cos = torch.cos(torch.deg2rad(true_position[:, :, 1]))

                pred_az_sin = torch.sin(torch.deg2rad(pred_position[:, :, 0]))
                pred_az_cos = torch.cos(torch.deg2rad(pred_position[:, :, 0]))
                pred_alt_sin = torch.sin(torch.deg2rad(pred_position[:, :, 1]))
                pred_alt_cos = torch.cos(torch.deg2rad(pred_position[:, :, 1]))

                true_positions = torch.stack([true_az_sin, true_az_cos, true_alt_sin, true_alt_cos], dim=2)
                pred_positions = torch.stack([pred_az_sin, pred_az_cos, pred_alt_sin, pred_alt_cos], dim=2)

                print("Actual")
                print(true_positions[0])
                print("Predicted")
                print(pred_positions[0])
                print("")

                pred_positions_by_planet = [pred_positions[:, 2 * i : 2 * i + 2]  for i in range(args.planet)]
                true_positions_by_planet = [true_positions[:, 2 * i : 2 * i + 2]  for i in range(args.planet)]


                loss_by_planet = [F.mse_loss(pred_positions_by_planet[i], true_positions_by_planet[i], reduction='sum') for i in range(self.args.planet)]

                size += batch['time'].shape[0] ## Used to normalize loss when plotting
                loss = [loss_by_planet[i] + loss[i] for i in range(self.args.planet)]

        return [(loss[i] / size).item() for i in range(self.args.planet)]


    def compute_loss(self, batch):
        batch_size = batch['time'].shape[0]

        ### Move data to GPU if available ###
        times = batch['time'].to(self.device)
        true_position = batch['pos'].to(self.device)
        ### Run forward ###
        pred_position = self.forward(times)

        # B x planets x 2

        # B x planets x 4 -> sin(alt), cos(alt), sin(az), cos(az)
        true = []
        pred = []
        true_az_sin = torch.sin(torch.deg2rad(true_position[:, :, 0]))
        true_az_cos = torch.cos(torch.deg2rad(true_position[:, :, 0]))
        true_alt_sin = torch.sin(torch.deg2rad(true_position[:, :, 1]))
        true_alt_cos = torch.cos(torch.deg2rad(true_position[:, :, 1]))

        pred_az_sin = torch.sin(torch.deg2rad(pred_position[:, :, 0]))
        pred_az_cos = torch.cos(torch.deg2rad(pred_position[:, :, 0]))
        pred_alt_sin = torch.sin(torch.deg2rad(pred_position[:, :, 1]))
        pred_alt_cos = torch.cos(torch.deg2rad(pred_position[:, :, 1]))

        true_positions = torch.stack([true_az_sin, true_az_cos, true_alt_sin, true_alt_cos], dim=2)
        pred_positions = torch.stack([pred_az_sin, pred_az_cos, pred_alt_sin, pred_alt_cos], dim=2)

        ### Compute loss on results ###
        loss = F.mse_loss(pred_positions, true_positions, reduction='sum')
        return loss
