import torch
import torch.nn as nn

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
        self.dropout = 0.1
        self.MLP_CL1 = nn.Sequential(
            nn.Linear(200, 400),
            nn.PReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(400, 200)
        )

    def forward(self, seq, msk):
        if msk is None:
            _ = torch.mean(seq, 1)
            # out = self.MLP_CL1(_)
            return _
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

