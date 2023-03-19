import torch
from torch import nn

class DJconv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DJconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        #use xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        
    def forward(self, H, U, I):
        Hu = torch.concat((H, torch.matmul(H, torch.matmul(H.t(), H))), dim=1)
        Hu = torch.where(Hu >= 0.5, 1., 0.)
        Hu = Hu.to(torch.float32)
    
        Du_v = torch.sum(Hu, dim=1)
        mask = Du_v.nonzero()
        for i in mask:
            Du_v[i] = 1 / torch.sqrt(Du_v[i])
        Du_v = torch.diag(Du_v)
        Du_v = Du_v.to(torch.float32)

        Du_e = torch.sum(Hu, dim=0)
        mask = Du_e.nonzero()
        for i in mask:
            Du_e[i] = 1 / torch.sqrt(Du_e[i])
        Du_e = torch.diag(Du_e)
        Du_e = Du_e.to(torch.float32)
        
        U = U.to(torch.float32)
        M_u = torch.linalg.multi_dot([Du_v, Hu, Du_e, Du_e, Hu.t(), Du_v, U]) + U
        # print(M_u.isnan().any())

        Hi = torch.concat((H.t(), torch.matmul(H.t(), torch.matmul(H, H.t()))), dim=1)
        Hi = torch.where(abs(Hi)>=0.5, 1., 0.)
        Hi = Hi.to(torch.float32)

        Di_v = torch.sum(Hi, dim=1)
        mask = Di_v.nonzero()
        for i in mask:
            Di_v[i] = 1 / torch.sqrt(Di_v[i])
        Di_v = torch.diag(Di_v)
        Di_v = Di_v.to(torch.float32)

        Di_e = torch.sum(Hi, dim=0)
        mask = Di_e.nonzero()
        for i in mask:
            Di_e[i] = 1 / torch.sqrt(Di_e[i])
        Di_e = torch.diag(Di_e)
        Di_e = Di_e.to(torch.float32)

        I = I.to(torch.float32)
        M_i = torch.linalg.multi_dot([Di_v, Hi, Di_e, Di_e, Hi.t(), Di_v, I]) + I
        # print(M_i.isnan().any())

        U_out = torch.matmul(M_u, self.weight) + self.bias
        I_out = torch.matmul(M_i, self.weight) + self.bias

        return U_out, I_out

        