
import torch
import time
import torch.nn as nn 


from torch.utils.data import DataLoader
from dataset import Data
from torch.utils.tensorboard import SummaryWriter
from cycle_loss import CycleLoss
from models import Net
from parameters import *


train_dataset = Data(type="train")
val_dataset = Data(type="val")
test_dataset = Data(type="test")

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
writer = SummaryWriter('./runs/Net', 'Net')

model = Net(num_channels=num_channels, 
            in_channel=in_channel, 
            out_channel=out_channel, 
            padding=padding, 
            activation=activation,
            frame_size=frame_size, 
            num_layers=num_layers)

mse_loss = nn.MSELoss()
cycle_loss = CycleLoss(batch_size=batch_size)

if torch.cuda.is_available(): 
    model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for e in range(epochs):
    t0 = time.time()
    model.train()
    for i, (img, track) in enumerate(train_loader):
        img = img.float()
        track = track.float()
        if torch.cuda.is_available():
            img, track = img.cuda(), track.cuda()

        optimizer.zero_grad()
        target_track = model(img)
        cycle = cycle_loss(target_track, track)
        mse = mse_loss(target_track, track) 
        loss = mse + cycle

        loss.backward()
        optimizer.step()

        train_mse += mse.item()
        train_cycle += cycle.item()
               
        train_loss = train_mse + train_cycle
        writer.add_scalar('Training loss',
                        train_loss / len(train_loader),
                        e)

    model.eval()
    for i, (img_val, track_val) in enumerate(val_loader):
        img_val, track_val = img_val.float(), track_val.float()
        img_val, track_val = img_val.cuda(), track_val.cuda()

        target_track_val = model(img_val)
        cycle = cycle_loss(target_track_val, track)
        mse = mse_loss(target_track_val, track_val)
        loss = mse + cycle

        valid_mse += mse.item()
        valid_cycle += cycle.item()
        valid_loss = valid_mse + valid_cycle

    writer.add_scalar('Validation loss',
                        valid_loss / len(val_loader),
                        e)
    if min_valid_loss > valid_loss:
        min_valid_loss = valid_loss
