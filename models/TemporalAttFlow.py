import torch
import torch.nn as nn 
import torch.nn.functional as F

class TemporalAttentionOverFlow(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(TemporalAttentionOverFlow, self).__init__()
        self.linear1 = nn.Linear(dim*5, dim)
        self.linear2 = nn.Linear(dim, 1)
    
    def forward(self, decoder_state, flow_matrix):
        '''
            decoder_state.shape = 1, batch_size, d2
            flow_matrix.shape = batch_size, 20, 4*d2
        '''
        batch_size, t, d = flow_matrix.shape
        _, _, d0 = decoder_state.shape

        decoder_state = decoder_state.repeat(t, 1, 1)
        decoder_state = torch.transpose(decoder_state, 0, 1)
        inputs = torch.cat((flow_matrix, decoder_state), 2).view(-1, self.dim*5)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, t)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), flow_matrix)

        return context


