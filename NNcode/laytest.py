import torch
from torch import nn
import torch.nn.functional as F
import math


class MPOLinear3(nn.Module):
    def __init__(self, input_size, output_size, num_cores, rank):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_cores = num_cores
        self.rank = rank
        
        def is_perfect_root(size):
            root = size**(1/num_cores)
            return round(root)**self.num_cores == input_size

        assert is_perfect_root(self.input_size), "input_size**(1/num_cores) must be round"
        assert is_perfect_root(self.output_size), "output_size**(1/num_cores) must be round"

        self.in_leg_dim = math.ceil(input_size**(1/num_cores))
        self.out_leg_dim = math.ceil(output_size**(1/num_cores))

        first_core = torch.Tensor(self.in_leg_dim, rank, self.out_leg_dim)
        self.first_core = nn.Parameter(first_core)

        middle_cores = [torch.Tensor(self.in_leg_dim, rank, rank, self.out_leg_dim) for _ in range(num_cores - 2)]
        middle_cores = [nn.Parameter(middle_cores[i]) for i in range(len(middle_cores))]
        self.middle_cores = nn.ParameterList(middle_cores)

        last_core = torch.Tensor(self.in_leg_dim, rank, self.out_leg_dim)
        self.last_core = nn.Parameter(last_core)

        bias = torch.Tensor(self.output_size)
        self.bias = nn.Parameter(bias)

        bound = (1/math.sqrt(self.input_size))

        nn.init.uniform_(self.first_core, -bound, bound)

        for i in range(len(self.middle_cores)):
            nn.init.uniform_(self.middle_cores[i], -bound, bound)
        
        nn.init.uniform_(self.last_core, -bound, bound)
        
        nn.init.uniform_(self.bias, -bound, bound)
    def addlastcore(self,A,lastcore):
        A_shape = A.shape
        num_dims_A = len(A_shape)
        A_labels = ''.join(chr(ord('a') + i) for i in range(num_dims_A))

        res_label=A_labels[:-2] + A_labels[-1] + 'ik'
        sumstring=f'{A_labels},ijk->{res_label}'

        return torch.einsum(sumstring,A,lastcore)
    def addx(self,x,res):
        sumstring='iabc,axbycz->ixyz'
        return torch.einsum(sumstring,x,res)
    def forward(self,x): #only works for 3 cores
        x = x.view(-1, *([self.in_leg_dim]*self.num_cores))
        midcore=self.middle_cores[0]
        res=torch.einsum('ijk,abcd->ikacd',self.first_core,midcore)
        res=self.addlastcore(res,self.last_core)
        x=self.addx(x,res)
        x = x.view(-1, self.output_size)
        return x+self.bias

