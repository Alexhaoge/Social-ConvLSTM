import torch.nn as nn
import torch

from .downsample import DownSampleForLSTM

class SocialLSTM(nn.Module):
    """Social LSTM

    Args:

    Inputs:
        _input: input tensor should be in shape of
        (batch, channel, seq_len,  input_size, input_size).
        NOTE: currently all version will only take channel 0 only!
            
    Returns:
        output of LSTMs
        (seq_len, batch, lstm_num_square, lstm_num_square)
    """

    def __init__(self, input_size:int,
                lstm_num_square:int=3,
                layer_num:int=1,
                dropout_rate:float=0.,
                upsample:bool=False):
        super(SocialLSTM, self).__init__()
        self.input_size = input_size
        self.lstm_num_square = lstm_num_square
        # Downsample layer before LSTM
        self.downsample = DownSampleForLSTM(input_size, lstm_num_square)
        # Create lstm grids
        self.lstms = nn.ModuleList()
        for _ in range(lstm_num_square):
            lstm_row = nn.ModuleList()
            for _ in range(lstm_num_square):
                lstm_row.append(nn.LSTMCell(
                    input_size = self.downsample.version,
                    hidden_size = 2
                ))
            self.lstms.append(lstm_row)

        # Linear layer to embed the input position
        self.input_embed = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if torch.cuda.is_available():
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):
            #nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [int(nodeID) for nodeID in PedsList[framenum]]
            # List of nodes
            #print("lookup table :%s"% look_up)
            list_of_nodes = [look_up[x] for x in nodeIDs]
            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                corr_index = corr_index.cuda()
            #print(list_of_nodes.data)
            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            grid_current = grids[framenum]
            # Get the corresponding hidden and cell states
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)
            
            h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(h_nodes)

            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states
