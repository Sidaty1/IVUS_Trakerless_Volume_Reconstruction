import torch 
import torch.nn as nn 
import numpy as np 
from parameters import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvLSTMCell(nn.Module):
    """
        ConvLSTM cell
    """
    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)


        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )
        # Current Hidden State
        H = output_gate * self.activation(C)
        return H, C



class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, 
        kernel_size, padding, activation, frame_size)

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len, 
        height, width, device=device)
        
        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, 
        height, width, device=device)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, 
        height, width, device=device)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convLSTMcell(X[:,:,time_step], H, C)

            output[:,:,time_step] = H

        return output




class Seq2Vector(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, padding, 
    activation, frame_size, num_layers, features=features):

        super(Seq2Vector, self).__init__()

        self.sequential = nn.Sequential()

        # Add First layer (Different in_channels than the others)
        self.sequential.add_module(
            "convlstm1", ConvLSTM(
                in_channels=in_channel, out_channels=out_channel,
                kernel_size=kernel_size, padding=padding, 
                activation=activation, frame_size=frame_size)
        )

        self.sequential.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=out_channel)
        ) 

        # Add the rest of the layers
        for l in range(2, num_layers+1):

            self.sequential.add_module(
                f"convlstm{l}", ConvLSTM(
                    in_channels=out_channel, out_channels=out_channel,
                    kernel_size=kernel_size, padding=padding, 
                    activation=activation, frame_size=frame_size)
                )
                
            self.sequential.add_module(
                f"batchnorm{l}", nn.BatchNorm3d(num_features=out_channel)
                ) 
        
        # linear layers
        self.sequential.add_module("flat", nn.Flatten())        
        for i in range(3):
            self.sequential.add_module(f"lin{i}",nn.Linear(in_features=features[i], out_features=features[i+1]))
            self.sequential.add_module(f"act{i}",nn.ReLU()) 
        
        self.sequential.add_module("lin4",nn.Linear(in_features=features[-2], out_features=features[-1]))

    def forward(self, X):
        output = self.sequential(X)
        return output


class Net(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, 
    activation, frame_size, num_layers): 
        super().__init__()

        self.seqtovec = Seq2Vector(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, 
                                            padding=padding, activation=activation, frame_size=frame_size, num_layers=num_layers)


    def forward(self, x):
        x1 = torch.squeeze(x[:,0], dim=2)
        x2 = torch.squeeze(x[:,1], dim=2)
        x1_out = torch.unsqueeze(self.seqtovec(x1), dim=1)
        x2_out = torch.unsqueeze(self.seqtovec(x2), dim=1)
        return torch.cat([x1_out, x2_out], dim=1) 