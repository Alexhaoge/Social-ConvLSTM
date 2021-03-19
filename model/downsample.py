import torch
import torch.nn as nn
from torch import empty, cat

class DownSampleForLSTM(nn.Module):
    """Downsample Module before LSTM

    Downsample module before LSTM.

    Args:
        input_size: input tensor shape (B,C,Seq,H, W).
        lstms_shape: Number of lstms is 
                        ``lstms_shape[0] x lstms_shape[0]``.
                        Default 3.
        version: default 3.

    Inputs:
        _input: input tensor should be in shape of
        (batch, channel, seq_len, H, W).
        
    Ouputs:
        downsample temperature grid in shape of
        (seq_len, batch, lstms_shape[0], lstms_shape[1], version*channel)

    """
    def __init__(self, input_size, lstms_shape,
                version:int=3):
        super(DownSampleForLSTM, self).__init__()
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(lstms_shape, int) else lstms_shape
        self.output_channels = version * input_size[1]
        self.version = version
        self.pools = nn.ModuleList()
        self.pools.append(SelectCenter(input_size, self.lstms_shape))
        if version > 1:
            self.pools.append(nn.AdaptiveAvgPool2d(self.lstms_shape))
        if version > 2:
            self.pools.append(nn.AdaptiveMaxPool2d(self.lstms_shape))


    def forward(self, _input):
        # Compute each downsample tensor and 
        # concat them in the new axis 4
        for pool in self.pools:
            pool_out = pool(_input).permute(2,0,3,4,1)
            # print(pool)
            # print(pool_out.shape)
            output = cat([output, pool_out], axis=-1)
        return output


class SelectCenter(nn.Module):
    """Customized sampling module

    Just roughly select the center grid of each block as downsample

    Inputs:
        (..., input_size, input_size)

    Outputs:
        (..., lstms_shape[0], lstms_shape[1])
    """
    def __init__(self, input_size, lstms_shape):
        super(SelectCenter, self).__init__()
        # calculate the index to be select
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(lstms_shape, int) else lstms_shape
        grid_len_x = input_size[-2] // self.lstms_shape[0]
        self.grid_select_x = []
        for x in range(grid_len_x // 2, input_size[0], grid_len_x):
            self.grid_select_x.append(x)
        grid_len_y = input_size[-1] // self.lstms_shape[1]
        self.grid_select_y = []
        for y in range(grid_len_y // 2, input_size[0], grid_len_y):
            self.grid_select_y.append(y)
        self.grid_select_y = torch.tensor(self.grid_select_y)
        if torch.cuda.is_available():
            self.grid_select_x = self.grid_select_x.cuda()
            self.grid_select_y = self.grid_select_y.cuda()
            

    def forward(self, _input):
        _output = _input.index_select(-2, self.grid_select_x)
        return _output.index_select(-1, self.grid_select_y)
