import torch

from src.model.layers.encoder import Encoder


class Classifier(torch.nn.Module):
    def __init__(self, encoder: Encoder, n_classes: int, hidden_units: list[int]):

        super().__init__()

        self.encoder = encoder

        self.n_classes = n_classes
        self.hidden_units = hidden_units

        self.sequential_block = torch.nn.Sequential()

        prev_feat = self.encoder.embedding_size
        for n_units in hidden_units:
            self.sequential_block.append(torch.nn.Linear(prev_feat, n_units))
            self.sequential_block.append(torch.nn.ReLU())
            prev_feat = n_units

        self.output_layer = torch.nn.Linear(prev_feat, n_classes)

    def forward(self, embeddings, mask):

        sequence = self.encoder(embeddings, mask=mask)
        sequence = sequence * torch.unsqueeze(mask, dim=-1)

        sequence = torch.transpose(sequence, 1, 2)
        encoding, _ = torch.max(sequence, dim=2)

        encoding = self.sequential_block(encoding)

        output = self.output_layer(encoding)

        return output
