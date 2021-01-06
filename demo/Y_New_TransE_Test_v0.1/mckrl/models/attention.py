import torch
import torch.nn as nn

class Attention_out(nn.Module):
    # [16,32,64,128,256]
    def __init__(self, in_size, hidden_size=128):
        super(Attention_out, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
