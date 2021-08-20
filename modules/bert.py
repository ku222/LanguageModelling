
from typing import Tuple
from torch import nn, FloatTensor, LongTensor

from .embedding import EmbeddingLayer
from .encoder import EncoderLayer
from .attnmask import AttnMask
from .heads import NSPHead, MLMHead



class BertConfig(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.nlayers = 12
        self.nheads = 12
        self.nh = 768
        self.maxseqlen = 512
        self.vocabsize = 30522
        self.embedding_dropout = 0.1
        self.encoder_dropout = 0.1
        self.attention_dropout = 0.1

        for (varname, varvalue) in kwargs.items():
            setattr(self, varname, varvalue)


class BertEncoderBase(nn.Module):
    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.embedding = EmbeddingLayer(vocabsize=cfg.vocabsize, nh=cfg.nh, max_position_embeddings=cfg.maxseqlen, dropout=cfg.embedding_dropout)
        self.attnmask = AttnMask()
        self.encoderlayers = nn.ModuleList([
            EncoderLayer(
                nh=cfg.nh,
                nheads=cfg.nheads,
                attention_dropout=cfg.attention_dropout,
                encoder_dropout=cfg.encoder_dropout)
            for _ in range(cfg.nlayers)
        ])

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, token_type_ids: LongTensor = None, position_ids: LongTensor = None) -> Tuple[FloatTensor, FloatTensor]:
        x = self.embedding(input_ids, token_type_ids, position_ids)
        square_mask = self.attnmask(attention_mask)
        for layer in self.encoderlayers:
            (x, square_mask) = layer(x, square_mask)
        return (x, square_mask)


class BertPreTraining(nn.Module):
    def __init__(self, cfg: BertConfig, pretrained_base: BertEncoderBase = None):
        super().__init__()
        self.encoder = pretrained_base if pretrained_base else BertEncoderBase(cfg=cfg)
        self.mlmhead = MLMHead(vocabsize=cfg.vocabsize, nh=cfg.nh)
        self.nsphead = NSPHead(nh=cfg.nh)

    def forward(self, input_ids: LongTensor, attention_mask: LongTensor, token_type_ids: LongTensor = None, position_ids: LongTensor = None) -> Tuple[FloatTensor, FloatTensor]:
        (encoder_out, _) = self.encoder.forward(input_ids, attention_mask, token_type_ids, position_ids)    # (batch, seqlen, nh)
        mlm_logits = self.mlmhead(encoder_out)                                                              # (batch, seqlen, vocabsize)
        nsp_logits = self.nsphead(encoder_out)                                                              # (batch, 2)
        return (mlm_logits, nsp_logits)
