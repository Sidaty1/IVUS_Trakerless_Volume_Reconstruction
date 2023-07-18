import torch
import torch.nn as nn
import numpy as np

from utils import *

class CycleLoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.mse = nn.MSELoss()
        self.batch_size = batch_size
    
    def reshape(self, input):
        """
            reshape (batch, transformations ) -> (batch, translations, rotations)
        """
        reshaped = torch.empty((input.shape[0], 2, int(input.shape[1]/2)))
        reshaped = torch.zeros_like(reshaped)
        for j in range(input.shape[0]):
            for i in range(0, int(input.shape[1]/2), 3):
                reshaped[j][0][i:i+3] = input[j][i*2:i*2+3]
                reshaped[j][1][i:i+3] = input[j][i*2+3:i*2+6]
        return reshaped

    def get_cycles(self, input, batch_size):

        global_pred_paths = []
        for batch in range(batch_size): 
            pred_paths = [] 
            batch_pred_t    = input[batch][0] # translation
            batch_pred_rot  = input[batch][1] # rotation
            for i in range(10):
                path_pred   = [batch_pred_t[0:3].detach().numpy(), euler_to_matrix(batch_pred_rot[3:6].detach().numpy())]
                for j in range(i): 
                    path_pred[0] += batch_pred_t[j*3:j*3+3].detach().numpy()
                    path_pred[1] *= euler_to_matrix(batch_pred_rot[j*3:j*3+3].detach().numpy())
                path_pred = [path_pred[0].tolist(), matrix_to_euler(path_pred[1]).tolist()]
                
                pred_paths.append(path_pred)
            global_pred_paths.append(pred_paths)
        return torch.from_numpy(np.asarray(global_pred_paths))

    def forward(self, pred, gt):
        pred = self.reshape(pred)
        gt = self.reshape(gt)

        pred_cycles = self.get_cycles(pred, pred.shape[0])
        gt_cycles = self.get_cycles(gt, gt.shape[0])

        loss = self.mse(pred_cycles, gt_cycles)
        loss /= self.batch_size

        return loss



            
