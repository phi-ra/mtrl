import torch.nn as nn
import torch.nn.functional as F

class BaseMLP(nn.Module):
    def __init__(self, arch,
                 input_size=2,
                 output_size = 1,
                 dropout=0.1, 
                 output_function=None):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self._set_arch(arch, input_size)
        self.output_function = output_function
        
    def _set_arch(self, arch, input_size):
        current_size = input_size
        for lay_size in arch:
            self.layers.append(nn.Linear(current_size, lay_size))
            current_size = lay_size
            
        self.final_layer = nn.Linear(current_size, self.output_size)

    def forward(self, x):
        for lay_ in self.layers:
            x = F.relu(lay_(x))
            x = self.dropout(x)
            
        x = self.final_layer(x)

        if self.output_function is not None:
            x = self.output_function(x)
        
        return x