import torch
from torch import nn
import torch.nn.functional as F
import tensornetwork as tn
import math

class StandardLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        weights = torch.Tensor(output_size, input_size)
        self.weights = nn.Parameter(weights)
        
        bias = torch.Tensor(output_size)
        self.bias = nn.Parameter(bias)

        bound = 1/math.sqrt(input_size)
        nn.init.uniform_(self.weights, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return F.linear(x, self.weights, self.bias)
    
class MPOLinear(nn.Module):
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
    
    def forward(self, x):
        x = x.view(-1, *([self.in_leg_dim]*self.num_cores))
        x_node = tn.Node(x, name="input node", backend="pytorch")

        core_nodes = [tn.Node(self.first_core, name="first core", backend="pytorch")]
        for i in range(len(self.middle_cores)):
            core_nodes.append(tn.Node(self.middle_cores[i], name=f"middle core {i}", backend="pytorch"))
        core_nodes.append(tn.Node(self.last_core, name="last core", backend="pytorch"))

        x_node[1] ^ core_nodes[0][0]
        core_nodes[0][1] ^ core_nodes[1][1]
        for i in range(1, len(core_nodes) - 1):
            x_node[i + 1] ^ core_nodes[i][0]
            core_nodes[i][2] ^ core_nodes[i + 1][1]
        x_node[-1] ^ core_nodes[-1][0]

        temp = x_node @ core_nodes[0]
        for i in range(1, len(core_nodes)):
            temp = temp @ core_nodes[i]
        
        x = temp.tensor.view(-1, self.output_size)

        return x + self.bias

class MPOLinear2(nn.Module):
    def __init__(self, input_shape, output_shape, ranks):
        def prod(factors):
            if factors:
                return factors[-1]*prod(factors[:-1])
            return 1
        
        super().__init__()
        self.input_shape = input_shape
        self.input_size = prod(self.input_shape)
        self.output_shape = output_shape
        self.output_size = prod(self.output_shape)

        first_core = torch.Tensor(input_shape[0], ranks[0], output_shape[0])
        self.first_core = nn.Parameter(first_core)
        middle_cores = [torch.Tensor(input_shape[i], ranks[i - 1], ranks[i], output_shape[i]) for i in range(1, len(input_shape) - 1)]
        middle_cores = [nn.Parameter(middle_cores[i]) for i in range(len(middle_cores))]
        self.middle_cores = nn.ParameterList(middle_cores)
        last_core = torch.Tensor(input_shape[-1], ranks[-1], output_shape[-1])
        self.last_core = nn.Parameter(last_core)
        bias = torch.Tensor(self.output_size)
        self.bias = nn.Parameter(bias)

        bound = 1/math.sqrt(self.input_size)
        nn.init.uniform_(self.first_core, -bound, bound)
        for i in range(len(self.middle_cores)):
            nn.init.uniform_(self.middle_cores[i], -bound, bound)
        nn.init.uniform_(self.last_core, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        x_node = tn.Node(x, name="input node", backend="pytorch")
        core_nodes = [tn.Node(self.first_core, name="first core", backend="pytorch")]
        for i in range(len(self.middle_cores)):
            core_nodes.append(tn.Node(self.middle_cores[i], name=f"middle core {i}", backend="pytorch"))
        core_nodes.append(tn.Node(self.last_core, name="last core", backend="pytorch"))

        x_node[1] ^ core_nodes[0][0]
        core_nodes[0][1] ^ core_nodes[1][1]
        for i in range(1, len(core_nodes) - 1):
            x_node[i + 1] ^ core_nodes[i][0]
            core_nodes[i][2] ^ core_nodes[i + 1][1]
        x_node[-1] ^ core_nodes[-1][0]

        temp = x_node @ core_nodes[0]
        for i in range(1, len(core_nodes)):
            temp = temp @ core_nodes[i]
        
        x = temp.tensor.view(-1, self.output_size)

        return x + self.bias