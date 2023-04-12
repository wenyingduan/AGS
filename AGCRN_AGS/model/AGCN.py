import torch
import torch.nn.functional as F
import torch.nn as nn


    
class StaticL0AVWGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(StaticL0AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.layernorm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
       
        # no adapative node
        #self.weights_pool = nn.Parameter(torch.FloatTensor(N, cheb_k, dim_in, dim_out))
        #self.bias_pool = nn.Parameter(torch.FloatTensor(N, dim_out))
    def forward(self, x, node_embeddings, mask=None):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        node_embeddings = self.dropout(self.layernorm(node_embeddings))
                             #N, dim_out
       
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        if mask is not None:
            supports = mask*supports
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv
    
    
class AVWGCN_(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(AVWGCN, self).__init__()
        
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.linear = nn.Linear(dim_in, embed_dim, bias = False)
        self.embed_dim = embed_dim
       
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [B, N, D] -> supports shaped [B,N, N]
        #output shape [B, N, C]
        
        x_ =self.linear(x)
        
        #print(x.size())
        #supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        ada_embeddings = torch.cat([node_embeddings.expand(x.size(0),170, self.embed_dim),x_],dim=-1)
        supports = F.softmax(F.relu(torch.bmm(ada_embeddings, ada_embeddings.transpose(-1, -2))), dim=1) # B,N,N
        support_set = [torch.eye(170).unsqueeze(0).expand(x.size(0), 170, 170).to(supports.device), supports]
        #print(supports.size())
        #ada_supports = F.softmax(F.relu(torch.bmm(x_, x_.transpose(-1, -2))), dim=1)
        #support_set = [torch.eye(node_num).unsqueeze(0).expand(x.size(0), node_num, node_num).to(supports.device), ada_supports*supports.unsqueeze(0)]
        #support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 2
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0) #(K,B,N,N)
        
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
       
        bias = torch.matmul(node_embeddings, self.bias_pool)   #N, dim_out
        
        
        x_g = torch.einsum("kbnm,bmc->kbnc", supports, x)      #cheb_k,B, N, dim_in
    
        x_g = x_g.permute(1, 2, 0, 3)  # B, N, cheb_k, dim_in
        
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        #print(x_gconv.size())
        #l1_supports = torch.norm(supports.transpose(0,1).contiguous().view(x.size(0),-1),p=1, dim =-1, keepdim = False)                        # L1 norm
        return x_gconv, supports
    
    
    

import math
from torch.nn.modules import Module

from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


class L0AWVGCN(Module):
    """Implementation of L0 regularization for AGCN"""
    def __init__(self, node_nums, dim_in, dim_out, cheb_k, embed_dim, bias=False, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0AWVGCN, self).__init__()
        self.node_num = node_nums
        self.prior_prec = weight_decay
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        #self.weights = Parameter(torch.Tensor(node_nums, node_nums))
        #self.node_embeddings = Parameter(torch.Tensor(node_nums, embed_dim)) #  generate the adjacency matrix A in function 'sample_weights'
        self.qz_loga = nn.Parameter(torch.Tensor(node_nums, node_nums)) # location for the adjacency matrix A
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.local_rep = local_rep
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
       

    def reset_parameters(self):
        #init.kaiming_normal_(self.node_embeddings, mode='fan_out')

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

    def regularization(self, supports):
        return self._reg_w(supports)

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

    def sample_z(self, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(self.node_num, self.node_num))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = torch.sigmoid(self.qz_loga)
            
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)
            
    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.node_num,self.node_num)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        #return mask.view(self.in_features, 1) * self.weights
        return mask

    def forward(self, x, node_embeddings): # x:[B,N,D]
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #[N,K,I,O]
        bias = torch.matmul(node_embeddings, self.bias_pool) ##[N,O]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1) #[N,N]
        '''
        if self.local_rep or not self.training:
            support_set = [torch.eye(self.node_num).to(supports.device), supports] #[[N,N],[N,N]]
            for k in range(2, self.cheb_k):
                support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
            supports = torch.stack(support_set, dim =0) #  [K,N,N]
            z = self.sample_z(x.size(0), sample=self.training) #[B,N,N]
            xin = torch.einsum("bni,bnn->bni",x,z) #[B, N, I], [B,N,N]->[B,N,I]
            x_g = torch.einsum("knm,bmi->bkni", supports,xin) # [B,N,N],[B,N,I]->[B,N,I]
           
        else:
        '''
        mask = self.sample_weights()
        supports = mask*supports # [N,N],[N,N]->[N,N]
        support_set = [torch.eye(self.node_num).to(supports.device), supports] #[[N,N],[N,N]]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim =0) #  [K,N,N]
        #supports = mask.unsqueeze(0)*supports # add mask to 单位矩阵
        x_g = torch.einsum("knm,bmi->bkni", supports, x) #[N,N],[B,N,I]->[B,N,I]
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, K, I
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        return x_gconv
    
    def __repr__(self):
        s = ('{name}({in_features} -> {out_features}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
             'local_rep={local_rep}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
    


    
   
