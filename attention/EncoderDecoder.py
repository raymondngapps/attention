import torch.nn as nn

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    
    It is the two Nx blocks in the diagram.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """_summary_

        Args:
            memory (_type_): The q, k vectors from the encoder
            src_mask (_type_): Use in training to mask out padding tokens
            tgt (_type_): The target sequence
            tgt_mask (_type_): Use in training to mask out padding tokens

        Returns:
            _type_: _description_
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)