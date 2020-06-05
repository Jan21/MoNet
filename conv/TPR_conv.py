import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from .inits import reset, glorot, zeros
from torch_geometric.utils import degree
import torch.nn.functional as F

EPS = 1e-15


class TPRConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 TPR_filler_dim,
                 dim,
                 bias=True,
                 reduce_for_TPR = False,
                 add_nonlin_before_TPR = False,
                 do_TPR_in_update = False,
                 **kwargs):
        super(TPRConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.do_TPR_in_update = do_TPR_in_update
        self.reduce_for_TPR = reduce_for_TPR
        self.add_nonlin_before_TPR = add_nonlin_before_TPR
        self.TPR_filler_dim = TPR_filler_dim

        self.lin = torch.nn.Linear(self.in_channels,
                                   self.out_channels,
                                   bias=False)
        self.lin_beforeTPR = torch.nn.Linear(self.out_channels,
                                   self.TPR_filler_dim,
                                   bias=False)
        self.reduce_final_TPR = torch.nn.Linear(self.out_channels*self.out_channels,
                                   self.out_channels*2,
                                   bias=False)

        if not self.reduce_for_TPR:
            self.TPR_filler_dim = out_channels

        self.linTPRreduction = torch.nn.Linear(3*4*self.TPR_filler_dim,
                                   self.out_channels,
                                   bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels*2))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        reset(self.lin)
        reset(self.linTPRreduction)
        reset(self.reduce_final_TPR)
        reset(self.lin_beforeTPR)

    def forward(self, x, edge_index, pseudo):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        out = self.lin(x)
        if self.add_nonlin_before_TPR:
            out = F.elu(out)
        fillers = None
        if self.reduce_for_TPR:
            fillers = self.lin_beforeTPR(out)
            out = self.propagate(edge_index, x=fillers, pseudo=pseudo,x_in=x,non_reduced=out)
        else: out = self.propagate(edge_index, x=out, pseudo=pseudo,x_in=x,non_reduced=out)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, pseudo,x_in,x):
        #(E, D), K = pseudo.size(), self.mu.size(0)

        #gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D))**2 # u každého vrcholu, vypočti jeho vzdálenost k 25 středním hodnotám na 2 osach
        #gaussian = gaussian / (EPS + self.sigma.view(1, K, D)**2)
        #gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1] # for each example it will have probability under each of the 25 gaussians

        #return (x_j * gaussian).sum(dim=1)

        num_of_messages = len(x_j)
        distances = pseudo[:,0]
        angles = pseudo[:,1]
        dist_mask = distances > 0.7
        dist_c1 = (0.7 - distances)/0.7
        dist_c2 = (1 - distances)/0.34
        dist_c1_comp = 1 - dist_c1
        dist_c2_comp = 1 - dist_c2
        dist_f1 = (~dist_mask) * dist_c1
        dist_f2 = dist_mask * dist_c2 + (~dist_mask) * dist_c1_comp
        dist_f3 = dist_mask * dist_c2_comp
        
        dist_roles = torch.stack([dist_f1,dist_f2,dist_f3],dim=1)
        
        i1 = angles < 0.2
        i2 = (angles >= 0.2) & (angles < 0.4)
        i3 = (angles >= 0.4) & (angles < 0.6)
        i4 = (angles >= 0.6) & (angles < 0.8)
        i5 = angles >= 0.8

        c1 = ((0.2 - angles)/0.2)*0.5
        c1_comp = 1 - c1
        c2 = (0.4 - angles)/0.2
        c2_comp = 1 - c2

        c3 = (0.6 - angles)/0.2
        c3_comp = 1 - c3
        c4 = (0.8 - angles)/0.2
        c4_comp = 1 - c4

        c5_comp = ((angles-0.8)/0.2)*0.5
        c5 = 1 - c5_comp

        f1 = i1*c1_comp + i2*c2 + i5*c5_comp
        f2 = i2*c2_comp + i3*c3
        f3 = i3*c3_comp + i4*c4
        f4 = i4*c4_comp + i5*c5 + i1*c1
        angle_roles = torch.stack([f1,f2,f3,f4],dim=1)
        TPR_messages = torch.einsum('bi,bj,bk->bijk', (dist_roles, angle_roles, x_j)).view(num_of_messages,-1)

        return TPR_messages


    def update(self, inputs,x_in,x,non_reduced):
        num_nodes = len(x)
        reduced_TPR = self.linTPRreduction(inputs)
        if self.reduce_for_TPR:
            x = non_reduced
        if self.do_TPR_in_update:
                TPR_for_node = torch.einsum('bi,bj->bij', (reduced_TPR, non_reduced)).view(num_nodes, -1)
                reduced_TPR_for_node = self.reduce_final_TPR(TPR_for_node)
                return reduced_TPR_for_node
        return torch.cat([reduced_TPR,x],dim=1)

    def __repr__(self):
        return '{}({}, {}, kernel_size={})'.format(self.__class__.__name__,
                                                   self.in_channels,
                                                   self.out_channels,
                                                   self.kernel_size)
