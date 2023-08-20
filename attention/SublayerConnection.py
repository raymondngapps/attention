import torch.nn as nn
from LayerNorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    
    This is the add & norm plus the multi-head attention layer or the feed forward layer in the diagram.
    """

    def __init__(self, size: int, dropoutRate: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropoutRate)

    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size. 
        Drop out is also applied, to help training network with relative few samples.
        
        Args:
            sublayer (_type_): The sublayer is the multi-head attention layer or the feed forward layer.
            
        Return:
            _type_: Value calculate by sublayer plus the add & norm in the diagram. 
        
        """
        return x + self.dropout(sublayer(self.norm(x)))