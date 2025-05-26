from torch import nn
import torch

class Crisp(nn.Module) :
    def __init__(self) :
        super().__init__()
    
    def forward(self, input) :
        output = input + 1
        return output

crisp = Crisp()
x = torch.tensor(1.0)
output = crisp(x)
print(output)
        