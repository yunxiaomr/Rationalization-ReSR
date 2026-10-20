from enum import Enum
from abc import ABC, abstractmethod

import torch
import numpy as np


class Encoder(ABC):
    def __init__(
        self,
        initial_embedding_size: int,
        embedding_size: int,
        max_len: int,
        apply_positional_encoding: bool,
    ):

        self.initial_embedding_size = initial_embedding_size
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.apply_positional_encoding = apply_positional_encoding

        if self.apply_positional_encoding:
            self.positional_encoding = self._get_encoding()
        else:
            self.positional_encoding = None

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def _get_angles(self, pos, i):

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / self.initial_embedding_size)
        return pos * angle_rates

    def _get_encoding(self):

        angle_rads = self._get_angles(
            np.arange(self.max_len)[:, np.newaxis],
            np.arange(self.initial_embedding_size)[np.newaxis, :],
        )

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return pos_encoding


class RecurrentEncoder(torch.nn.Module, Encoder):
    def __init__(
        self,
        initial_embedding_size: int,
        embedding_size: int,
        max_len: int,
        apply_positional_encoding: bool,
        dropout: float,
        bidirectional: bool = False,
    ):

        torch.nn.Module.__init__(self)
        Encoder.__init__(
            self,
            initial_embedding_size,
            embedding_size,
            max_len,
            apply_positional_encoding,
        )

        assert 0.0 <= dropout < 1.0

        self.dropout = dropout
        self.bidirectional = bidirectional

        hidden_size = embedding_size if not bidirectional else embedding_size // 2

        self.gru = torch.nn.GRU(
            input_size=initial_embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, embeddings, mask):

        if self.positional_encoding is not None:
            embeddings += self.positional_encoding

        embeddings = embeddings * torch.unsqueeze(mask, dim=-1)

        seq, final_state = self.gru(embeddings)

        return seq


class AttentionEncoder(torch.nn.Module, Encoder):
    def __init__(
        self,
        initial_embedding_size: int,
        embedding_size: int,
        max_len: int,
        apply_positional_encoding: bool,
        dropout: float,
        n_heads: int = 1,
    ):

        torch.nn.Module.__init__(self)
        Encoder.__init__(
            self,
            initial_embedding_size,
            embedding_size,
            max_len,
            apply_positional_encoding,
        )

        assert 0.0 <= dropout < 1.0

        self.initial_embedding_size = initial_embedding_size
        self.embedding_size = embedding_size
        self.max_len = max_len
        self.dropout = dropout
        self.n_heads = n_heads
        self.apply_positional_encoding = apply_positional_encoding

        self.self_attention_layer = torch.nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def _build_attention_mask(self, mask):

        mask_a = torch.unsqueeze(mask, dim=2)
        mask_b = torch.unsqueeze(mask, dim=1)

        attention_mask = torch.matmul(mask_a, mask_b)

        return attention_mask

    def forward(self, embeddings, mask):

        attention_mask = self._build_attention_mask(mask)

        if self.positional_encoding is not None:
            embeddings += self.positional_encoding

        attention_output = self.self_attention_layer(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=attention_mask,
            need_weights=False,
            attn_mask=attention_mask,
        )

        return attention_output


class EncoderType(Enum):
    RECURRENT = 1
    ATTENTION = 2


class EncoderBuilder:
    def __init__(self, encoder_type: EncoderType):
        self.encoder_type = encoder_type

    def instantiate(self, params: dict) -> Encoder:

        if self.encoder_type == EncoderType.RECURRENT:
            return RecurrentEncoder(**params)

        elif self.encoder_type == EncoderType.ATTENTION:
            return AttentionEncoder(**params)

        else:
            raise Exception("Invalid Encoder Type:", self.encoder_type)
