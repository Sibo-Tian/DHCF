import torch
from torch import nn
from layers import DJconv

class DHCF(nn.Module):
    def __init__(self, U_dim, I_dim, dropout, embedding_dim=64, num_layers=1) -> None:
        super(DHCF, self).__init__()
        self.embed_U = nn.Linear(U_dim, embedding_dim)
        self.embed_I = nn.Linear(I_dim, embedding_dim)

        self.layers = [DJconv(embedding_dim, embedding_dim) for _ in range(num_layers)]
        self.dropout = [nn.Dropout(dropout) for _ in range(num_layers)]
    
    def forward(self, H, U, I):
        U = self.embed_U(U)
        I = self.embed_I(I)
        U_out = U.clone()
        I_out = I.clone()
        for idx, layer in enumerate(self.layers):
            U = self.dropout[idx](U)
            I = self.dropout[idx](I)
            U, I = layer(H, U, I)
            U_out = torch.concat((U_out, U), dim=1)
            I_out = torch.concat((I_out, I), dim=1)
        return U_out, I_out


