
import torch
from torch import nn, FloatTensor, LongTensor
import math


class SelfAttentionLayer(nn.Module):
    """
    Sub-block module to compute self-attention
    """
    def __init__(self, nh: int, nheads: int, dropout=0.1):
        super().__init__()
        self.nheads = nheads
        self.nh = nh

        self.wq = nn.Linear(in_features=nh, out_features=nheads * nh)
        self.wk = nn.Linear(in_features=nh, out_features=nheads * nh)
        self.wv = nn.Linear(in_features=nh, out_features=nheads * nh)
        self.wz = nn.Linear(in_features=nheads * nh, out_features=nh)
        self.dropout = nn.Dropout(dropout)

    def scoring_transpose(self, QKV: FloatTensor):
        """
        Args:
            QKV: FloatTensor of shape (batch, seqlen, nheads * nh)

        Returns:
            QKV reshaped to (batch, nheads, seqlen, nh)
        """
        (batch, seqlen, _) = QKV.shape
        QKV = QKV.view(batch, seqlen, self.nheads, self.nh)
        return QKV.permute(0, 2, 1, 3)

    def forward(self, x: FloatTensor, attnmask: FloatTensor) -> FloatTensor:
        """
        Performs forward pass through encoder block
            - input [x] -> multiply with [wq], [wk], [wv] to produce multi-headed [Q], [K], [V] matrices
            - [Q] multiply with [K.T] transposed to produce [QK_t], softmax to get scores
            - [QK_t] multiply with [V] to produce matrix [Z], which is our attention context vector
            - Condense the multi-headed [Z] matrix into a single matrix, same as input shape by multiplying with [wz]

        Args:
            x: Token encodings or self-attention outputs of previous Encoder block, shape of (batch, seqlen, embdim)
            attnmask: square attention Mask, shape of (batch, 1, seqlen, seqlen)

        Returns:
            Self-attention-encoded outputs, same shape of (batch, seqlen, embdim)
        """
        Q = self.wq(x)                                                  # (batch, seqlen, nheads * nh)
        K = self.wk(x)                                                  # (batch, seqlen, nheads * nh)
        V = self.wv(x)                                                  # (batch, seqlen, nheads * nh)
        Q, K, V = [self.scoring_transpose(qkv) for qkv in (Q, K, V)]    # Convert all to shape of (batch, nheads, seqlen, nh)

        (batch, nheads, seqlen, nh) = Q.shape

        QK_t = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(nh)     # (batch, nheads, seqlen, nh) mm (batch, nheads, nh, seqlen) -> (batch, nheads, seqlen, seqlen)
        QK_t = QK_t + attnmask                                          # (batch, nheads, seqlen, seqlen) + (batch, 1, seqlen, seqlen) -> (batch, nheads, seqlen, seqlen)
        QK_t = torch.softmax(QK_t, dim=-1)                              # (batch, nheads, seqlen, seqlen)
        QK_t = self.dropout(QK_t)                                       # Unusual that this drops out entire tokens to attend to, but done by original paper.

        Z = torch.matmul(QK_t, V)                                       # (batch, nheads, seqlen, seqlen) mm (batch, nheads, seqlen, nh) -> (batch, nheads, seqlen, nh)
        Z = Z.permute(0, 2, 1, 3).contiguous()                          # (batch, seqlen, nheads, nh)
        Z = Z.view(batch, seqlen, nheads * nh)                          # (batch, seqlen, nheads * nh) concatenate the heads
        return self.wz(Z)                                               # (batch, seqlen, nheads * nh) mm (nheads * nqkv, embdim) -> (batch, seqlen, embdim)
