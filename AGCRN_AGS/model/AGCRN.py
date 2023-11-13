import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init
import math
from model.AGCRNCell import AGCRNCell, AGLSTMCell
limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


class AVWDCLSTM(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCLSTM, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGLSTMCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGLSTMCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, mask):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        #assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        #b =x.size(0)
        #embed_dim = node_embeddings.size(-1)
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            state = (state[:,0],state[:,1])
            inner_states = []
            for t in range(seq_length):
                
                state= self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings, mask)
                
                inner_states.append(state[0])
                
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, 2, B, N, hidden_dim)


class AVWDCRNN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1, dropout_on = True):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings, mask):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings,mask)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class AGCRN(nn.Module):
    def __init__(self, args, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False):
        super(AGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.scale = args.scale
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        self.epochs = args.epochs
        self.l0epochs = args.l0epochs
        #L0 norm
        self.prior_prec = weight_decay
        #self.qz_loga = nn.Parameter(torch.Tensor(self.num_node, self.num_node)) # location for the adjacency matrix A
        self.qz_loga = None
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        #L0 norm
        
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        #self.x_embed = nn.Linear(args.input_dim, args.x_embed_dim)
        #self.encoder = AVWDCLSTM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim, 
                                # args.num_layers)
        self.qz_project = nn.Linear(args.embed_dim, self.num_node)
        self.encoder = AVWDCRNN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k, args.embed_dim, 
                                 args.num_layers)

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
    def reset_parameters(self):
        #init.kaiming_normal_(self.node_embeddings, mode='fan_out')
        if self.qz_loga is not None:
            self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)
        
    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self, node_embeddings):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        logpw_col = torch.sum(- (.5 * self.prior_prec * supports.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 
        return logpw + logpb
    
    def regularization(self):
        return - (1. / (self.scale)*self._reg_w(self.node_embeddings))

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.node_num
        expected_l0 = ppos * self.node_num
        
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps
            
    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.num_node,self.num_node)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        #return mask.view(self.in_features, 1) * self.weights
        return mask

    def forward(self, source, targets, epoch,teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        #source = self.x_embed(source)
        mask = None
        #mask =torch.zeros(170,170).to(source.device)
        if self.l0epochs+self.epochs>=epoch>self.epochs:
            self.qz_loga = self.qz_project(self.node_embeddings)
            mask = self.sample_weights()
            #mask =torch.zeros(170,170).to(source.device)
        #elif 750>epoch>=550:
            #mask = self.sample_weights()
            #mask = self.mask_dropout(mask)
        #elif epoch>=self.l0epochs:
            #mask = torch.zeros(self.num_node,self.num_node).to(source.device)
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, mask)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        
        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        
        #output = torch.mean(output, dim=2)
        #output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim)
        return output, mask
    def test(self, source, targets, mask,teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        #source = self.x_embed(source)
        #mask = self.sample_weights()
        #mask = torch.zeros(self.num_node,self.num_node).to(source.device)
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, mask)      #B, T, N, hidden
        output = output[:, -1:, :, :]                                   #B, 1, N, hidden

        #CNN based predictor
        output = self.end_conv((output))                         #B, T*C, N, 1
        
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C
        
        #output = torch.mean(output, dim=2)
        #output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim)
       
        return output,mask
    '''
    def regularization(self):
        regularization = 0
        for cell in self.encoder.dcrnn_cells:
            regularization += - (1. / self.scale) * cell.gcn.regularization(self.node_embeddings)
        return regularization
    '''
    
        
    def l1_norm(self):
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        return torch.linalg.norm(supports,1)
    
    def return_supports(self):
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        return supports
    def return_states(self):
        dict_states ={}
        dict_states['weights_pool']= self.encoder.dcrnn_cells[1].gcn.weights_pool.detach().cpu()
        dict_states['node_embeddings'] = self.node_embeddings.detach().cpu()
        dict_states['supports'] = self.return_supports()
        return dict_states
        
    def __repr__(self):
        s = ('{name}({in_features} -> {out_features}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
             'local_rep={local_rep}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    

    

     

