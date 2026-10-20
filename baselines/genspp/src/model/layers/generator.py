import torch
from src.model.layers.encoder import Encoder


class Generator(torch.nn.Module):
    def __init__(self, encoder: Encoder, hidden_units: list[int]):

        super().__init__()

        self.encoder = encoder
        self.hidden_units = hidden_units

        self.sequential_block = torch.nn.Sequential()

        prev_feat = self.encoder.embedding_size
        for n_units in hidden_units:
            self.sequential_block.append(torch.nn.Linear(prev_feat, n_units))
            self.sequential_block.append(torch.nn.ReLU())
            prev_feat = n_units

        self.output_layer = torch.nn.Linear(prev_feat, 1)
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, inputs, mask, use_hard_mask=True):

        embeddings = self.encoder(inputs, mask=mask)

        embeddings = self.sequential_block(embeddings)

        soft_mask = self.output_layer(embeddings)
        soft_mask = self.output_activation(soft_mask)

        soft_mask = torch.squeeze(soft_mask, dim=-1)

        if use_hard_mask:
            hard_mask = soft_mask + 0.5
            hard_mask = torch.floor(hard_mask)
            return hard_mask * mask, soft_mask * mask

        else:
            return soft_mask * mask
