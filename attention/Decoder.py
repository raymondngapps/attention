import torch.nn as nn
from Utils import clones
from LayerNorm import LayerNorm

class Decoder(nn.Module):
    """
    Generic N layer decoder with masking,  the layer contains three sub-layers, incluing the feed forward layer, 
    the mask multi-head attention layer, multi=head attention and the add & norm.   
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """_summary_

        Args:
            x (_type_): _description_
            memory (_type_): The q, k vectors from the encoder
            src_mask (_type_): _description_
            tgt_mask (_type_): _description_

        Returns:
            _type_: _description_
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)