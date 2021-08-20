

import torch
from torch import nn, FloatTensor
from torch.nn.functional import gelu


class NSPHead(nn.Module):
    def __init__(self, nh=768):
        super().__init__()
        self.ff_pooler = nn.Linear(in_features=nh, out_features=nh)
        self.ff_logits = nn.Linear(in_features=nh, out_features=2)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Args:
            x: Output from last Encoder layer - shape of (batch, seqlen, nh)

        Returns:
            pooled logits for binary prediction task, shape of (batch, 2)
        """
        firsttoken = x[:, 0, :]                             # (batch, nh)
        pooled = torch.tanh(self.ff_pooler(firsttoken))     # (batch, nh)
        return self.ff_logits(pooled)                       # (batch, 2)


class MLMHead(nn.Module):
    def __init__(self, vocabsize=30522, nh=768):
        super().__init__()
        self.ff_transform = nn.Linear(in_features=nh, out_features=nh)
        self.layernorm = nn.LayerNorm(nh)
        self.ff_decoder = nn.Linear(in_features=nh, out_features=vocabsize)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.layernorm(gelu(self.ff_transform(x)))      # (batch, seqlen, nh)
        return self.ff_decoder(x)                           # (batch, seqlen, vocabsize)
