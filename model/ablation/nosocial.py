import torch.nn as nn
from model.downsample import DownSampleForLSTM
import torch

class VanillaLSTM_Downsample(nn.Module):
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
        (seq_len, batch, lstms_shape[0], lstms_shape[1])
    """
    def __init__(
        self, input_size, lstms_shape=3,
        embedding_size: int = 32,
        hidden_size: int = 32,
        layer_num: int = 1,
        dropout_rate: float = 0.5,
        downsample_version: int = 3,
        step: int = 5,
        *args, **kwargs
    ):
        super(VanillaLSTM_Downsample, self).__init__()
        self.input_size = input_size
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(lstms_shape, int) else lstms_shape
        self.hidden_size = hidden_size
        # Downsample layer before LSTM
        self.downsample = DownSampleForLSTM(
            input_size, self.lstms_shape, downsample_version)
        # Create lstm grids
        self.lstms = nn.ModuleList()
        for _ in range(self.lstms_shape[0]):
            lstm_row = nn.ModuleList()
            for _ in range(self.lstms_shape[1]):
                lstm_row.append(nn.LSTM(
                    input_size = embedding_size,
                    hidden_size = hidden_size,
                    num_layers = layer_num,
                    dropout=dropout_rate))
            self.lstms.append(lstm_row)
        # LINEAR embedding layer after downsample
        self.down_linear_embed = nn.Linear(downsample_version, embedding_size)
        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        # time fusion
        self.conv3d = nn.Conv3d(input_size[2], step, 1)
        # output FC layer
        self.out_linear = nn.Linear(hidden_size, 1)
        

    def forward(self, _input):
        downsample_out = self.downsample(_input)
        output_shape = list(downsample_out.shape)
        output_shape[-1] = self.hidden_size
        output = torch.empty(output_shape)
        if torch.cuda.is_available():
            output = output.cuda()
        for i in range(self.lstms_shape[0]):
            for j in range(self.lstms_shape[1]):
                # embed the output of downsample
                _x = self.dropout(self.relu(
                    self.down_linear_embed(downsample_out[:, :, i, j, :])))
                # throw the embeded _x into lstm
                output[:, :, i, j, :] = self.lstms[i][j](_x)[0]
        output_fusion = self.dropout(self.relu(
            self.conv3d(output.permute(1,0,2,3,4))
        )).permute(1,0,2,3,4)
        return self.out_linear(output_fusion).squeeze()


