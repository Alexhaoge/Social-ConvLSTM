import torch
import torch.nn as nn
from model.downsample import SelectCenter

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


class RMSEDownSample(nn.Module):
    def __init__(self, input_size, lstm_num_square=3, eps=1e-6):
        super().__init__()
        self.rmse = RMSELoss(eps)
        self.down = SelectCenter(input_size, lstm_num_square)
    
    def forward(self, yhat, y):
        return self.rmse(yhat, self.down(torch.squeeze(y).permute(1,0,2,3)))


class L1LossDownSample(nn.Module):
    def __init__(self, input_size, lstm_num_square=3):
        super().__init__()
        self.l1loss = nn.L1Loss()
        self.down = SelectCenter(input_size, lstm_num_square)
    
    def forward(self, yhat, y):
        return self.l1loss(yhat, self.down(torch.squeeze(y).permute(1,0,2,3)))