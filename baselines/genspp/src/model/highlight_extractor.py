import torch

from src.model.layers.encoder import EncoderBuilder
from src.model.layers.generator import Generator
from src.model.layers.classifier import Classifier


class HighlightExtractor(torch.nn.Module):
    def __init__(
        self,
        n_classes: int,
        generator_hidden_units: list[int],
        classifier_hidden_units: list[int],
        gen_encoder_builder: EncoderBuilder,
        cl_encoder_builder: EncoderBuilder,
        gen_encoder_params: dict,
        cl_encoder_params: dict,
        shared_encoder: bool = False,
    ):

        super().__init__()

        self.n_classes = n_classes
        self.generator_hidden_units = generator_hidden_units
        self.classifier_hidden_units = classifier_hidden_units
        self.gen_encoder_builder = gen_encoder_builder
        self.cl_encoder_builder = cl_encoder_builder
        self.gen_encoder_params = gen_encoder_params
        self.cl_encoder_params = cl_encoder_params
        self.shared_encoder = shared_encoder

        if shared_encoder:
            if gen_encoder_builder.encoder_type != cl_encoder_builder.encoder_type:
                print(
                    "Different encoder specified for Generator and Classifier, using the Generator one"
                )
            encoder = gen_encoder_builder.instantiate(gen_encoder_params)
            self.generator = Generator(encoder, generator_hidden_units)
            self.classifier = Classifier(encoder, n_classes, classifier_hidden_units)

        else:
            encoder_1 = gen_encoder_builder.instantiate(gen_encoder_params)
            encoder_2 = cl_encoder_builder.instantiate(cl_encoder_params)
            self.generator = Generator(encoder_1, generator_hidden_units)
            self.classifier = Classifier(encoder_2, n_classes, classifier_hidden_units)

        self.generator.requires_grad_(False)

    def forward(self, embeddings, mask):

        highlight_mask, soft_mask = self.generator(embeddings, mask=mask)
        classification_output = self.classifier(embeddings, mask=highlight_mask)

        return classification_output, highlight_mask, soft_mask
