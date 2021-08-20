from typing import Tuple
from torch import nn, FloatTensor
from torch.nn.functional import gelu

from .selfattention import SelfAttentionLayer


class EncoderLayer(nn.Module):
    """
    Encoder block module to compute self-attention
    """
    def __init__(self, nh=768, nh_intermediate=3072, nheads=12, attention_dropout=0.1, encoder_dropout=0.1):
        super().__init__()
        self.selfattn = SelfAttentionLayer(nh=nh, nheads=nheads, dropout=attention_dropout)
        self.ff1 = nn.Linear(in_features=nh, out_features=nh_intermediate)
        self.ff2 = nn.Linear(in_features=nh_intermediate, out_features=nh)
        self.drop = nn.Dropout(p=encoder_dropout)
        self.norm1 = nn.LayerNorm(nh)
        self.norm2 = nn.LayerNorm(nh)

    def forward(self, x: FloatTensor, attnmask: FloatTensor) ->  Tuple[FloatTensor, FloatTensor]:
        """
        Args:
            x: Token encodings or self-attention outputs of previous Encoder block, shape of (batch, seqlen, nh)

        Returns:
            Normalised self-attention-encoded outputs, same shape of (batch, seqlen, nh)
        """
        attn_out = self.selfattn(x, attnmask)       # (batch, seqlen, nh)

        # Add and normalise input + attention output
        attn_out = self.norm1(x + attn_out)         # (batch, seqlen, nh)

        # Intermediate feedforward + activation
        ff1_out = gelu(self.ff1(attn_out))          # (batch, seqlen, nh_intermediate)

        # Output feedforward + dropout applied
        ff2_out = self.drop(self.ff2(ff1_out))      # (batch, seqlen, nh)

        # Add and normalise attn output and dense output
        final_out = self.norm2(attn_out + ff2_out)
        return (final_out, attnmask)
