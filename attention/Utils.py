import torch.nn as nn
import copy
import torch

def clones(module: nn.Module, N: int):
    """
    Produce N identical layers by doing deep copies of the module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size: int):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    
    "triu - Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0."
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    
    """
    Return a diagonal matrix with true or false values, for example if size = 5, then the result will be:
    tensor([[[ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]]])
    """
    return subsequent_mask == 0