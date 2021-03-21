import torch
from torch import nn
from .convlstm import ConvLSTMCell

class SocialConvLSTM(nn.Module):
    """
    input (B, C, T, H, W)
    output (B, 1, T, H, W)
    """
    def __init__(self,
        input_size, num_layers, hidden_dim,
        kernel_size, device, dropout_rate,
        step: int = 5,
        lstms_shape = 2,
        share: bool = False,
        embed_dim: int = 1,
        *args, **kwargs
    ):
        super(SocialConvLSTM, self).__init__()
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(lstms_shape, int) else lstms_shape
        self.share = share
        self.step = step
        self.device = device
        self.image_size = input_size[-2:]
        self.lenx = self.image_size[0] // self.lstms_shape[0]
        self.leny = self.image_size[1] // self.lstms_shape[1]
        # print(self.image_size, self.lstms_shape, self.lenx, self.leny)
        self.cell = SocialConvLSTMCell(
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=(kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size,
            dropout_rate=dropout_rate,
            embed_dim=embed_dim,
            lstms_shape=self.lstms_shape,
            share=share
        )
        self.conv_time = nn.Conv3d(
            in_channels=input_size[2],
            out_channels=step,
            kernel_size=1
        )
        self.conv_dim = nn.Conv3d(
            in_channels=hidden_dim, 
            out_channels=1, 
            kernel_size=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, inputs: torch.Tensor):
        cur_state = self.cell.init_hidden_map(
            batch_size=inputs.shape[0],
            image_size=(self.lenx, self.leny)
        )
        out_raw_shape = list(inputs.shape)
        out_raw_shape[1] = self.conv_dim.in_channels
        out_raw = torch.empty(out_raw_shape, device=self.device)
        for t in range(inputs.shape[2]):
            input_map = {}
            for i, x in enumerate(range(0, inputs.shape[-2], self.lenx)):
                for j, y in enumerate(range(0, inputs.shape[-1], self.leny)):
                    input_map[str((i,j))] = inputs[:,:,t,x:x+self.lenx,y:y+self.leny]
            out_map = self.cell(input_map, cur_state)
            out_t = torch.cat([
                torch.cat(
                    [ out_map[0][str((i, j))] for j in range(self.lstms_shape[1])],
                -1) for i in range(self.lstms_shape[0])
            ], -2) # (B, hidden_dim, H, W)
            out_raw[:,:,t,:,:] = out_t
        out_fushion = self.dropout(self.relu(
            self.conv_time(out_raw.permute(0,2,1,3,4))
        )).permute(0,2,1,3,4)
        return self.conv_dim(out_fushion)            


class SocialConvLSTMCell(nn.Module):
    """
    input: dict{ str((i,j)): hidden state of (i,j) in shape (b, hidden_dim, h, w)}
    h = image_size[0] / lstms_shape[0], w = W / lstms_shape[1]
    output: tuple(h, c)
        h: dict{ str((i,j)): hidden state of (i,j) in shape (b, hidden_dim, h, w)}
        c: dict{ str((i,j)): cell state of (i,j) in shape (b, hidden_dim, h, w)}
    """
    def __init__(self,
        input_size, hidden_dim,
        kernel_size, dropout_rate,
        embed_dim: int = 1,
        lstms_shape = 2,
        share: bool = False,
        *args, **kwargs
    ):
        super(SocialConvLSTMCell, self).__init__()
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(lstms_shape, int) else lstms_shape
        self.share = share
        self.image_size = input_size[-2:]
        input_dim = input_size[1]
        assert self.image_size[0] % self.lstms_shape[0] == 0
        assert self.image_size[1] % self.lstms_shape[1] == 0
        if self.share:
            self.lstm = ConvLSTMCell(input_dim+embed_dim, hidden_dim, kernel_size, True)
        else:
            grid = []
            for i in range(self.lstms_shape[0]):
                for j in range(self.lstms_shape[1]):
                    grid.append((i,j))
            self.lstm = nn.ModuleDict({
                str(k): ConvLSTMCell(input_dim+embed_dim, hidden_dim, kernel_size, True) for k in grid
            })
        self.SCE = SocialConvEmbed(
            self.image_size, kernel_size, 
            hidden_dim, embed_dim, self.lstms_shape,
            dropout_rate)
        
    
    def forward(self, inputs: dict, cur_state: tuple):
        hidden_cat = torch.cat([
            torch.cat(
                [ cur_state[0][str((i, j))] for j in range(self.lstms_shape[1])],
                -1
            ) for i in range(self.lstms_shape[0])
        ], -2)
        social = self.SCE(hidden_cat)
        hidden_map = {}
        cell_map = {}
        # print(inputs)
        for i in range(self.lstms_shape[0]):
            for j in range(self.lstms_shape[1]):
                _cur = (cur_state[0][str((i,j))], cur_state[1][str((i,j))])
                _in = torch.cat([inputs[str((i,j))], social], dim=1)
                _h, _c = self.get_lstm(i, j).forward(_in, _cur)
                hidden_map[str((i,j))] = _h
                cell_map[str((i,j))] = _c
        return (hidden_map, cell_map)


    def get_lstm(self, x: int, y: int) -> ConvLSTMCell:
        return self.lstm if self.share else self.lstm[str((x,y))]

    def init_hidden_map(self, batch_size, image_size):
        hidden_map = {}
        cell_map = {}
        for i in range(self.lstms_shape[0]):
            for j in range(self.lstms_shape[0]):
                hidden_map[str((i,j))], cell_map[str((i,j))] = self.get_lstm(i,j).init_hidden(batch_size, image_size)
        return (hidden_map, cell_map)


class SocialConvEmbed(nn.Module):
    """
    hidden_cat (B, hidden_dim, H, W) -> social tensor (B, embed_dim, h, w)
    """
    def __init__(self, 
        image_size, kernel_size, hidden_dim, embed_dim, lstms_shape, dropout_rate):
        super(SocialConvEmbed, self).__init__()
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(lstms_shape, int) else lstms_shape
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(hidden_dim, embed_dim, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.avapool = nn.AdaptiveAvgPool2d((
            image_size[0] // self.lstms_shape[0],
            image_size[1] // self.lstms_shape[1]
        ))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embed = self.dropout(self.relu(self.conv(inputs)))
        return self.avapool(embed)