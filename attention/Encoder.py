import torch.nn as nn
from Utils import clones
from LayerNorm import LayerNorm
from EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers, the layer contains two sub-layers, incluing the feed forward layer, 
    the multi-head attention layer and the add & norm.
    """

    def __init__(self, layer: EncoderLayer, N: int):
        """
        This is the block that encapulates the encoder layers

        Args:
            layer (_type_): One signal layer of the encoder
            N (int): The number of repetitions of the layer
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn, then output is normalized before password to another Encoder.
        
        We employ a residual connection (https://arxiv.org/abs/1607.06450) around each of the two sub-layers, 
        followed by layer normalization (https://arxiv.org/abs/1607.06450), this significantly reduces the training time in feed-forward neural networks.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)