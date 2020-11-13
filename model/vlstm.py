from torch import tensor
import torch.nn as nn
from .downsample import DownSampleForLSTM
import torch

class VanillaLSTM(nn.Module):
    """Vanilla LSTMs

    Simple vanilla LSTMs for baseline. After downsample, 
    Each LSTM is responsible for a block and runs independently.

    Args:

    Inputs:
        _input: input tensor should be in shape of
        (batch, channel, seq_len,  input_size, input_size).
        NOTE: currently all version will only take channel 0 only!
        
    Returns:
        output of LSTMs
        (seq_len, batch, lstm_num_square, lstm_num_square)
    """
    def __init__(self,
                input_size:int,
                lstm_num_square:int=3,
                layer_num:int=1,
                dropout_rate:float=0.,
                upsample:bool=False):
        super(VanillaLSTM, self).__init__()
        self.input_size = input_size
        self.lstm_num_square = lstm_num_square
        # Downsample layer before LSTM
        self.downsample = DownSampleForLSTM(input_size, lstm_num_square)
        # Create lstm grids
        self.lstms = nn.ModuleList()
        for _ in range(lstm_num_square):
            lstm_row = nn.ModuleList()
            for _ in range(lstm_num_square):
                lstm_row.append(nn.LSTM(
                    input_size = self.downsample.version,
                    hidden_size = 1,
                    num_layers = layer_num,
                    dropout=dropout_rate))
            self.lstms.append(lstm_row)
        

    def forward(self, _input):
        downsample_out = self.downsample(_input)
        output_shape = list(downsample_out.shape)
        output_shape.pop()
        output = torch.empty(output_shape)
        if torch.cuda.is_available():
            output = output.cuda()
        for i in range(self.lstm_num_square):
            for j in range(self.lstm_num_square):
                _tmp = self.lstms[i][j](downsample_out[:, :, i, j, :])[0]
                output[:, :, i, j] = _tmp.squeeze()
        return output