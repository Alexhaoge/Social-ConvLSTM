import torch
from torch import nn
from torch.nn.modules.conv import Conv2d
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
                 lstms_shape=2,
                 share: bool = False,
                 embed_dim: int = None,
                 return_all_layers: bool = False,
                 *args, **kwargs
                 ):
        super(SocialConvLSTM, self).__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        embed_dim = self._extend_for_multilayer(embed_dim, num_layers)
        if not len(kernel_size) == len(self.hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(
            lstms_shape, int) else lstms_shape
        self.share = share
        self.step = step
        self.device = device
        self.image_size = input_size[-2:]
        self.lenx = self.image_size[0] // self.lstms_shape[0]
        self.leny = self.image_size[1] // self.lstms_shape[1]
        self.num_layers = num_layers
        self.return_all_layers = return_all_layers
        # print(self.image_size, self.lstms_shape, self.lenx, self.leny)

        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            _input_size = list(input_size)
            if i >= 1:
                _input_size[1] = self.hidden_dim[i-1]
            self.cell_list.append(
                SocialConvLSTMCell(
                    input_size=_input_size,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=kernel_size[i],
                    dropout_rate=dropout_rate,
                    embed_dim=embed_dim[i],
                    lstms_shape=self.lstms_shape,
                    share=share
                )
            )
        self.out_conv_time = nn.Conv3d(
            in_channels=input_size[2],
            out_channels=step,
            kernel_size=1
        )
        self.out_conv_dim = nn.Conv3d(
            in_channels=self.hidden_dim[-1],
            out_channels=1,
            kernel_size=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor, init_state: list = None) -> torch.Tensor:
        if init_state is None:
            init_state = self._init_hidden(
                batch_size=inputs.shape[0],
                image_size=(self.lenx, self.leny)
            )
        elif len(init_state) != self.num_layers:
            raise ValueError('init state size not match with layer number')
        out_raw_shape = list(inputs.shape)
        out_raw_shape[1] = self.hidden_dim[-1]
        out_raw = torch.empty(out_raw_shape, device=self.device)
        last_state_list = []  # h,c at time t for each layer
        input_list = []  # split input at time t
        for t in range(inputs.shape[2]):
            input_map = {}
            for i, x in enumerate(range(0, inputs.shape[-2], self.lenx)):
                for j, y in enumerate(range(0, inputs.shape[-1], self.leny)):
                    input_map[str((i, j))] = inputs[:, :, t,
                                                    x:x+self.lenx, y:y+self.leny]
            input_list.append(input_map)

        for l in range(self.num_layers):
            cur_state = init_state[l]
            for t in range(inputs.shape[2]):
                cur_state = self.cell_list[l](input_list[t], cur_state)
                input_list[t] = {}
                for i in range(self.lstms_shape[0]):
                    for j in range(self.lstms_shape[1]):
                        input_list[t][str((i,j))] = self.dropout(cur_state[0][str((i,j))])
                if l == self.num_layers - 1:
                    out_t = torch.cat([
                        torch.cat(
                            [cur_state[0][str((i, j))]
                             for j in range(self.lstms_shape[1])],
                            -1) for i in range(self.lstms_shape[0])
                    ], -2)  # (B, hidden_dim, H, W)
                    out_raw[:, :, t, :, :] = out_t
            last_state_list.append(cur_state)

        out_fushion = self.dropout(self.relu(
            self.out_conv_time(out_raw.permute(0, 2, 1, 3, 4))
        )).permute(0, 2, 1, 3, 4)
        return self.out_conv_dim(out_fushion)

    def _init_hidden(self, batch_size, image_size) -> list:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden_map(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size) -> None:
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers) -> list:
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


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
                 embed_dim: int = None,
                 lstms_shape=2,
                 share: bool = False,
                 *args, **kwargs
                 ):
        super(SocialConvLSTMCell, self).__init__()
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(
            lstms_shape, int) else lstms_shape
        self.share = share
        self.image_size = input_size[-2:]
        input_dim = input_size[1]
        assert self.image_size[0] % self.lstms_shape[0] == 0
        assert self.image_size[1] % self.lstms_shape[1] == 0
        if embed_dim is None:
            embed_dim = hidden_dim
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        grid = []
        for i in range(self.lstms_shape[0]):
            for j in range(self.lstms_shape[1]):
                grid.append((i, j))
        if self.share:
            self.lstm = ConvLSTMCell(
                input_dim+embed_dim, hidden_dim, kernel_size, True)
            # self.input_embed = Conv2d(
            #     input_dim, embed_dim, kernel_size, padding=padding)
        else:
            self.lstm = nn.ModuleDict({
                str(k): ConvLSTMCell(input_dim+embed_dim, hidden_dim, kernel_size, True) for k in grid
            })
            # self.input_embed = nn.ModuleDict({
            #     str(k): Conv2d(input_dim, embed_dim, kernel_size, padding=padding) for k in grid
            # })
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout_rate)
        self.SCE = SocialConvEmbed(
            self.image_size, kernel_size,
            hidden_dim, embed_dim, self.lstms_shape,
            dropout_rate)

    def forward(self, inputs: dict, cur_state: tuple):
        hidden_cat = torch.cat([
            torch.cat(
                [cur_state[0][str((i, j))]
                 for j in range(self.lstms_shape[1])],
                -1
            ) for i in range(self.lstms_shape[0])
        ], -2)
        social = self.SCE(hidden_cat)
        hidden_map = {}
        cell_map = {}
        # print(inputs)
        for i in range(self.lstms_shape[0]):
            for j in range(self.lstms_shape[1]):
                _cur = (cur_state[0][str((i, j))], cur_state[1][str((i, j))])
                # embed = self.dropout(self.relu(
                #     self.get_input_embed(i, j).forward(inputs[str((i, j))])
                # ))
                # _in = torch.cat([embed, social], dim=1)
                _in = torch.cat([inputs[str((i, j))], social], dim=1)
                _h, _c = self.get_lstm(i, j).forward(_in, _cur)
                hidden_map[str((i, j))] = _h
                cell_map[str((i, j))] = _c
        return hidden_map, cell_map

    def get_lstm(self, x: int, y: int) -> ConvLSTMCell:
        return self.lstm if self.share else self.lstm[str((x, y))]

    def get_input_embed(self, x: int, y: int) -> nn.Conv2d:
        return self.input_embed if self.share else self.input_embed[str((x, y))]

    def init_hidden_map(self, batch_size, image_size) -> tuple:
        hidden_map = {}
        cell_map = {}
        for i in range(self.lstms_shape[0]):
            for j in range(self.lstms_shape[0]):
                hidden_map[str((i, j))], cell_map[str((i, j))] = self.get_lstm(
                    i, j).init_hidden(batch_size, image_size)
        return (hidden_map, cell_map)


class SocialConvEmbed(nn.Module):
    """
    hidden_cat (B, hidden_dim, H, W) -> social tensor (B, embed_dim, h, w)
    """

    def __init__(self,
                 image_size, kernel_size, hidden_dim, embed_dim, lstms_shape, dropout_rate):
        super(SocialConvEmbed, self).__init__()
        self.lstms_shape = (lstms_shape, lstms_shape) if isinstance(
            lstms_shape, int) else lstms_shape
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(hidden_dim, embed_dim,
                              kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.avapool = nn.AdaptiveAvgPool2d((
            image_size[0] // self.lstms_shape[0],
            image_size[1] // self.lstms_shape[1]
        ))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embed = self.dropout(self.relu(self.conv(inputs)))
        return self.avapool(embed)
