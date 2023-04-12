import torch
import torch.nn as nn
from model.AGCN import StaticL0AVWGCN


    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = StaticL0AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = StaticL0AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings, mask=None):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings, mask))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings, mask))
        h = r*state + (1-r)*hc
       
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
'''    
class AGLSTMCell_(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGLSTMCell_, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gcn= AVWGCN(dim_in+self.hidden_dim, 3*dim_out, cheb_k, embed_dim)
        self.gate = AVWGCN(dim_in+self.hidden_dim, 1*dim_out, cheb_k, embed_dim)
    def forward(self, x, state, node_embeddings, mask):
        h,c = state
        batch_size = x.size(0)
       
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        h = h.to(x.device)
        c = c.to(x.device)
        
        input_and_state = torch.cat((x, h), dim=-1)
        
        ifg=self.gcn(input_and_state, node_embeddings) 
        o = self.gate(input_and_state, node_embeddings) 
        #g = self.gate(input_and_state, node_embeddings)
        i,f,g = torch.chunk(ifg,3,-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f*c +i*g
        new_h = o*torch.tanh(new_c)
        return (new_h,new_c)

    def init_hidden_state(self, batch_size):
        state =torch.zeros(batch_size, 2, self.node_num, self.hidden_dim)
        
        return state
 '''  
class AGLSTMCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGLSTMCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gcn= StaticL0AVWGCN(dim_in+self.hidden_dim, 4*dim_out, cheb_k, embed_dim)
     
    def forward(self, x, state, node_embeddings, mask):
        h,c = state
        batch_size = x.size(0)
       
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        h = h.to(x.device)
        c = c.to(x.device)
        
        input_and_state = torch.cat((x, h), dim=-1)
        
        ifgo=self.gcn(input_and_state, node_embeddings, mask) 
        #o = self.gate(input_and_state, node_embeddings) 
       
        i,f,g,o = torch.chunk(ifgo,4,-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        new_c = f*c +i*g
        new_h = o*torch.tanh(new_c)
        return (new_h,new_c)

    def init_hidden_state(self, batch_size):
        state =torch.zeros(batch_size, 2, self.node_num, self.hidden_dim)
        
        return state
    
    
