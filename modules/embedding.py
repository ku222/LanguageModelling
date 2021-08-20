import torch
import torch.nn as nn
from torch import nn, FloatTensor, LongTensor


class EmbeddingLayer(nn.Module):
    def __init__(self, vocabsize=30522, nh=768, dropout=0.1, max_position_embeddings=512):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=vocabsize, embedding_dim=nh)
        self.positional_embeddings = nn.Embedding(num_embeddings=max_position_embeddings, embedding_dim=nh)
        self.segment_embeddings = nn.Embedding(num_embeddings=2, embedding_dim=nh)
        self.layernorm = nn.LayerNorm(nh)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))   # (1, max_position_embeddings)

    def forward(self, input_ids: LongTensor, token_type_ids: LongTensor = None, position_ids: LongTensor = None) -> FloatTensor:
        """
        Args:
            Tokens:     shape of (batch, seqlen)
            Segments:   shape of (batch, seqlen)
        """
        (batch, seqlen) = input_ids.shape

        # No segments - assume all segment 0 (one sentence)
        if token_type_ids is None:
            token_type_ids = torch.zeros((batch, seqlen), dtype=torch.long, device=self.position_ids.device)

        # No positions - fill in range from 1 to max_position_embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :seqlen]

        embeddings = (
            self.token_embeddings(input_ids) +
            self.segment_embeddings(token_type_ids) +
            self.positional_embeddings(position_ids))

        return self.layernorm(self.dropout(embeddings))
