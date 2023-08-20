import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details). https://arxiv.org/abs/1512.03385
    
    This is the fully connected layer with value normalized to 0 mean and 1 std 
    (Feed Forward and Normalization in the diagram)
    """

    def __init__(self, features: int, eps: float=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): value come from (multi-head attention + add & norm)

        Returns:
            _type_: _description_
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2