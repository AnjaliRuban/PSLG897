import os
import sys
import random
import json
import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from planet_loader import PlanetDataset, collate_fn


class Module(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        ### Layers Here ###
        self.linear_theta1 = nn.Linear(1, 1)
        self.linear_radius1 = nn.Linear(1, 1)

        self.linear_manda = nn.Linear(1, 1)

        self.linear_phi2 = nn.Linear(1, args.planet)
        self.linear_theta2 = nn.Linear(1, args.planet)
        self.linear_radius2 = nn.Linear(1, args.planet)
        ### End Layers ###

        self.to(self.device)

    def forward(self, t):
        ### Models Here ###
        B = t.shape[0]

        theta_1 = self.linear_theta1(t) # -> B x 1
        radius_1 = self.linear_radius1(t) # -> B x 1

        mean_sun_x = torch.sin(thetha_1) * radius_1 # -> B x 1
        mean_sun_y = torch.cos(theta_1) * radius_1 # -> B x 1
        mean_sun_z = torch.tensor(0, dtype=torch.float).to(self.device) # -> B x 1

        manda_x = mean_sun_x.expand(B, args.planet)
        manda_y = mean_sun_y.expand(B, args.planet)
        manda_z = self.linear_manda(mean_sun_z).expand(B, args.planet)

        phi_2 = self.linear_phi2(torch.tensor(1).to(self.device))
        theta_2 = self.linear_theta2(t)
        radius_2 = self.linear_radius(torch.tensor(1).to(self.device))

        sighara_x = (radius_2 * torch.sin(phi_2) * torch.cos(theta_2)) + manda_x
        sighara_y = (radius_2 * torch.sin(phi_2) * torch.sin(theta_2)) + manda_y
        sighara_z = (radius_2 * torch.cos(phi_2)) + manda_z

        positions = self.convert_coordinates(sighara_x, sighara_y, sighara_z)

        return positions
        ### End Model ###

    def convert_coordinates(self, x, y, z):
        pass

    def run_train(self, data):
        ### Setup ###
        self.writer = SummaryWriter(args.writer) ## Our summary writer, which plots loss over time
        self.fsave = os.path.join(args.dout, 'best.pth') ## The path to the best version of our model

        with open(data['train'], 'r') as file: ## Train data loading
            train_data = json.load(file)
        train_dataset = PlanetDataset(train_data, self.args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

        with open(data['valid'], 'r') as file: ## Validation data loading
            valid_data = json.load(file)
        valid_dataset = PlanetDataset(valid_data, self.args)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

        optimizer = torch.optim.Adam(list(self.parameters()), lr=args.lr)

        best_loss = 1e10 ## This will hold the best validation loss so we don't overfit to training data

        ### Training Loop ###
        for epoch in range(args.epoch):
            print('Epoch', epoch)
            train_description = "Epoch " + str(epoch) + ", train"
            valid_description = "Epoch " + str(epoch) + ", valid"

            train_loss = torch.tensor(0, dtype=torch.float)
            train_size = torch.tensor(0, dtype=torch.float)

            self.train()

            for batch in tqdm.tqdm(train_dataloader, desc=train_description):
                optimizer.zero_grad() ## You don't want to accidentally have leftover gradients

                ### Compute loss and update variables ###
                loss = self.compute_loss(batch)

                train_size += batch['time'].shape[0] ## Used to normalize loss when plotting
                train_loss += loss

                ### Backpropogate ###
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            ### Run validation loop ###
            valid_loss, valid_size = self.run_eval(valid_dataloader, valid_description)

            ### Write to SummaryWriter ###
            self.writer.add_scalar("Loss/Train", str(train_loss/train_size), epoch)
            self.writer.add_scalar("Loss/Valid", str(valid_loss/valid_size), epoch)
            self.writer.flush()

            ### Save model if it is better that the previous ###
            if valid_loss < best_loss:
                print( "Obtained a new best validation loss of {:.2f}, saving model checkpoint to {}...".format(valid_loss, fsave))
                torch.save({
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args
                }, fsave)
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

        return size, loss

    def compute_loss(self, batch):
        batch_size = batch['time'].shape[0]

        ### Move data to GPU if available ###
        times = batch['time'].to(self.device)
        true_positions = batch['pos'].to(self.device)

        ### Run forward ###
        pred_positions = self.forward(times)

        ### Compute loss on results ###
        loss = F.mse_loss(pred_positions, true_positions)

        return loss
