import torch
import torch.nn as nn 
import torch.nn.functional as F

class AttentionFlow(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super(AttentionFlow, self).__init__()
        self.linear1 = nn.Linear(dim*3, 1)
    
    def forward(self, aud_feats, semantic_feats):
        '''
            aud_feats.shape = batch_size, 20, d2
            semantic_feats.shape = batch_size, 28, d2
        '''
        bs, t, d2 = aud_feats.shape
        _, n, _ = semantic_feats.shape
        
        S = torch.zeros(bs, t, n).cuda()

        for i in range(t):
            for j in range(n):
                h_t = aud_feats[:, i, :]
                h_n = semantic_feats[:, j, :]
                sim = torch.cat((h_t, h_n, h_t*h_n), dim=1)
                S[:, i, j] = self.linear1(sim).reshape(-1)
        # CALCULATE ATTENTION VALUES
        at = F.softmax(S, dim=2)
        b, _ = torch.max(S, dim=2)
        b = F.softmax(b, dim=1)

        # CALCULATE AUDIO CONTEXT VECTOR (h tilde a2)
        h_con_a2 = torch.bmm(b.unsqueeze(1), aud_feats) # b, 1, d2 (target2source)

        # CALCULATE SEMANTIC CONTEXT MATRIX (h tilde w)
        h_con_w = torch.bmm(at, semantic_feats) # b, t, d2 (source2target)

        flow_matrix = torch.cat((aud_feats, h_con_w, aud_feats*h_con_w,\
                aud_feats*h_con_a2), dim=2)

        return flow_matrix


