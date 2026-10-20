import torch as th
from modeling.layers import (
    SelectItem,
    AttentionPooling,
)
from torch.nn.functional import gumbel_softmax


class FR(th.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.0,
        embedding_matrix=None,
        freeze_embeddings=False,
        temperature=1.0,
    ):
        super().__init__()

        self.temperature = temperature

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.selection_head = selection_head()

        self.classification_head = classification_head()

        self.dropout = th.nn.Dropout(p=dropout_rate)

        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

    def generator(self, text, mask):

        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        tokens_emb, _ = self.encoder(tokens_emb)
        tokens_emb = self.layer_norm(tokens_emb)
        tokens_emb = self.dropout(tokens_emb)

        highlight_logits = self.selection_head(tokens_emb)
        highlight_hat = gumbel_softmax(logits=highlight_logits, tau=self.temperature, hard=True)[
            :, :, 1
        ]

        return highlight_hat, highlight_logits

    def classifier(self, text, mask, highlight_mask):

        hl_tokens_emb = self.embedding(text) * mask[:, :, None]
        hl_tokens_emb *= highlight_mask[:, :, None]

        hl_tokens_emb, _ = self.encoder(hl_tokens_emb)
        hl_tokens_emb = self.layer_norm(hl_tokens_emb)
        hl_tokens_emb = hl_tokens_emb * mask[:, :, None] + (1.0 - mask[:, :, None]) * (-1e6)
        hl_tokens_emb = th.transpose(hl_tokens_emb, 1, 2)

        hl_emb, _ = th.max(hl_tokens_emb, dim=2)
        hl_emb = self.dropout(hl_emb)

        logits = self.classification_head(hl_emb)

        return logits

    def forward(self, text, attention_mask, sample_ids):

        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        logits = self.classifier(text=text, mask=attention_mask, highlight_mask=highlight_hat)

        return logits, highlight_hat, highlight_logits, attention_mask


class MGR(th.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        num_generators,
        dropout_rate=0.0,
        embedding_matrix=None,
        freeze_embeddings=False,
        temperature=1.0,
    ):
        super().__init__()

        self.temperature = temperature

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)
        self.dropout = th.nn.Dropout(p=dropout_rate)

        self.num_generators = num_generators
        self.generators = th.nn.ModuleList()
        for gen_idx in range(num_generators):
            self.generators.append(
                th.nn.Sequential(
                    th.nn.GRU(
                        input_size=embedding_dim,
                        hidden_size=hidden_size,
                        batch_first=True,
                        bidirectional=True,
                    ),
                    SelectItem(0),
                    th.nn.LayerNorm(hidden_size * 2),
                    self.dropout,
                    selection_head(),
                )
            )

        self.classifier_encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.classification_head = classification_head()

    def generator(self, text, mask, generator_idx):

        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        highlight_logits = self.generators[generator_idx](tokens_emb)

        highlight_hat = gumbel_softmax(logits=highlight_logits, tau=self.temperature, hard=True)[
            :, :, 1
        ]

        return highlight_hat, highlight_logits

    def classifier(self, text, mask, highlight_mask):

        hl_tokens_emb = self.embedding(text) * mask[:, :, None]
        hl_tokens_emb *= highlight_mask[:, :, None]

        hl_tokens_emb, _ = self.classifier_encoder(hl_tokens_emb)
        hl_tokens_emb = self.layer_norm(hl_tokens_emb)
        hl_tokens_emb = hl_tokens_emb * mask[:, :, None] + (1.0 - mask[:, :, None]) * (-1e6)
        hl_tokens_emb = th.transpose(hl_tokens_emb, 1, 2)

        hl_emb, _ = th.max(hl_tokens_emb, dim=2)
        hl_emb = self.dropout(hl_emb)

        logits = self.classification_head(hl_emb)

        return logits

    def forward(self, text, attention_mask, sample_ids):
        hl_hat_list = []
        hl_logits_list = []
        logits_list = []

        for generator_idx in range(self.num_generators):
            gen_hl_hat, gen_hl_logits = self.generator(
                text=text, mask=attention_mask, generator_idx=generator_idx
            )
            hl_hat_list.append(gen_hl_hat)
            hl_logits_list.append(gen_hl_logits)

            logits = self.classifier(text=text, mask=attention_mask, highlight_mask=gen_hl_hat)
            logits_list.append(logits)

        return logits_list, hl_hat_list, hl_logits_list, attention_mask

    def forward_one_head(self, text, attention_mask, sample_ids):

        highlight_hat, highlight_logits = self.generator(
            text=text, mask=attention_mask, generator_idx=0
        )

        logits = self.classifier(text=text, mask=attention_mask, highlight_mask=highlight_hat)

        return logits, highlight_hat, attention_mask


class MCD(th.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.0,
        embedding_matrix=None,
        freeze_embeddings=False,
        temperature=1.0,
    ):
        super().__init__()

        self.temperature = temperature

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.gen_encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.cls_encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )

        self.classification_head = classification_head()

        self.gen_classification_head = selection_head()

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)
        self.gen_contextualizer = th.nn.Identity()

        self.gen = th.nn.ModuleList(
            [
                self.gen_encoder,
                self.layer_norm,
                self.dropout,
                self.gen_contextualizer,
                self.gen_classification_head,
            ]
        )

    def generator(self, text, mask):

        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        generator_hidden_states, _ = self.gen_encoder(tokens_emb)
        generator_hidden_states = self.layer_norm(generator_hidden_states)
        generator_hidden_states = self.dropout(generator_hidden_states)
        generator_hidden_states = self.gen_contextualizer(generator_hidden_states)

        highlight_logits = self.gen_classification_head(generator_hidden_states)
        highlight_hat = gumbel_softmax(logits=highlight_logits, tau=self.temperature, hard=True)[
            :, :, 1
        ]

        return highlight_hat, highlight_logits

    def classifier(self, text, mask, highlight_mask):

        hl_tokens_emb = self.embedding(text) * mask[:, :, None]
        hl_tokens_emb *= highlight_mask[:, :, None]

        hl_tokens_emb, _ = self.cls_encoder(hl_tokens_emb)
        hl_tokens_emb = hl_tokens_emb * mask[:, :, None] + (1.0 - mask[:, :, None]) * (-1e6)
        hl_tokens_emb = th.transpose(hl_tokens_emb, 1, 2)

        hl_emb, _ = th.max(hl_tokens_emb, dim=2)
        hl_emb = self.dropout(hl_emb)

        logits = self.classification_head(hl_emb)

        return logits

    def forward(self, text, attention_mask, sample_ids):

        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        logits = self.classifier(text=text, mask=attention_mask, highlight_mask=highlight_hat)

        return logits, highlight_hat, attention_mask

    def generator_forward(self, text, attention_mask, sample_ids):

        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        return highlight_hat, highlight_logits

    def classifier_forward(self, text, attention_mask, highlight_mask, sample_ids):

        logits = self.classifier(text=text, mask=attention_mask, highlight_mask=highlight_mask)
        return logits

    def no_selection_forward(self, text, attention_mask, sample_ids):

        logits = self.classifier(text=text, mask=attention_mask, highlight_mask=attention_mask)
        return logits

    def teacher_guidance_forward(
        self,
        text,
        attention_mask,
        sample_ids,
        target_labels=None,
        method: str = "margin",
    ):
        attention_mask = attention_mask.to(dtype=th.float32)
        if method not in {"margin", "grad_input"}:
            raise ValueError(f"Unsupported teacher guidance method: {method}")

        tokens_emb = self.embedding(text)
        tokens_emb = tokens_emb * attention_mask[:, :, None]

        if method == "grad_input":
            teacher_tokens_emb = tokens_emb.detach().clone().requires_grad_(True)
            cls_hidden_states, _ = self.cls_encoder(teacher_tokens_emb)
            masked_hidden_states = cls_hidden_states * attention_mask[:, :, None] + (
                1.0 - attention_mask[:, :, None]
            ) * (-1e6)
            pooled_hidden_states, _ = th.max(masked_hidden_states, dim=1)
            full_logits = self.classification_head(pooled_hidden_states)

            if target_labels is None:
                target_labels = th.argmax(full_logits, dim=-1)
            target_labels = target_labels.to(dtype=th.long)

            target_logits = full_logits.gather(dim=-1, index=target_labels[:, None]).sum()
            token_grads = th.autograd.grad(
                outputs=target_logits,
                inputs=teacher_tokens_emb,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            token_scores = (token_grads * teacher_tokens_emb).sum(dim=-1).abs()
            return full_logits, token_scores

        cls_hidden_states, _ = self.cls_encoder(tokens_emb)
        masked_hidden_states = cls_hidden_states * attention_mask[:, :, None] + (
            1.0 - attention_mask[:, :, None]
        ) * (-1e6)

        pooled_hidden_states, _ = th.max(masked_hidden_states, dim=1)
        full_logits = self.classification_head(pooled_hidden_states)

        if target_labels is None:
            target_labels = th.argmax(full_logits, dim=-1)
        target_labels = target_labels.to(dtype=th.long)

        token_logits = self.classification_head(cls_hidden_states)
        gather_index = target_labels[:, None, None].expand(-1, token_logits.size(1), 1)
        target_token_logits = token_logits.gather(dim=-1, index=gather_index).squeeze(-1)

        if token_logits.size(-1) > 1:
            target_mask = th.nn.functional.one_hot(
                target_labels,
                num_classes=token_logits.size(-1),
            ).to(dtype=th.bool)
            target_mask = target_mask[:, None, :].expand_as(token_logits)
            competitor_logits = token_logits.masked_fill(target_mask, float("-inf")).amax(dim=-1)
            competitor_logits = th.where(
                th.isfinite(competitor_logits),
                competitor_logits,
                th.zeros_like(competitor_logits),
            )
            token_scores = target_token_logits - competitor_logits
        else:
            token_scores = target_token_logits

        return full_logits, token_scores


class GRAT(th.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        classification_head,
        selection_head,
        dropout_rate=0.0,
        embedding_matrix=None,
        freeze_embeddings=False,
        temperature=1.0,
    ):
        super().__init__()

        self.temperature = temperature

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.gen_encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.cls_encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1,
            bidirectional=True,
        )

        self.classification_head = classification_head()

        self.gen_classification_head = selection_head()

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

        self.gen = th.nn.Sequential(
            self.gen_encoder,
            SelectItem(0),
            self.layer_norm,
            self.dropout,
            self.gen_classification_head,
        )

    def generator(self, text, mask):

        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        highlight_logits = self.gen(tokens_emb)
        highlight_hat = gumbel_softmax(logits=highlight_logits, tau=self.temperature, hard=True)[
            :, :, 1
        ]

        return highlight_hat, highlight_logits

    def classifier(self, text, mask, highlight_mask):

        hl_tokens_emb = self.embedding(text) * mask[:, :, None]
        hl_tokens_emb *= highlight_mask[:, :, None]

        hl_tokens_emb, _ = self.cls_encoder(hl_tokens_emb)
        hl_tokens_emb = self.layer_norm(hl_tokens_emb)
        hl_tokens_emb = hl_tokens_emb * mask[:, :, None] + (1.0 - mask[:, :, None]) * (-1e6)
        hl_tokens_emb = th.transpose(hl_tokens_emb, 1, 2)

        hl_emb, _ = th.max(hl_tokens_emb, dim=2)
        hl_emb = self.dropout(hl_emb)

        logits = self.classification_head(hl_emb)

        return logits

    def forward(self, text, attention_mask, sample_ids):

        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        logits = self.classifier(text=text, mask=attention_mask, highlight_mask=highlight_hat)

        return logits, highlight_hat, highlight_logits, attention_mask


class GRATGuider(th.nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        classification_head,
        noise_sigma=1.0,
        dropout_rate=0.0,
        embedding_matrix=None,
        freeze_embeddings=False,
        temperature=1.0,
    ):
        super().__init__()

        self.temperature = temperature
        self.noise_sigma = noise_sigma

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.activation = th.nn.GELU()
        self.convert_layer = th.nn.Linear(embedding_dim, hidden_size * 2, bias=True)
        self.encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_fc = th.nn.Sequential(
            th.nn.Linear(hidden_size * 2, hidden_size * 2),
            th.nn.GELU(),
            th.nn.Linear(hidden_size * 2, 1),
        )

        self.projection_fc = th.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.pooling = AttentionPooling(hidden_size * 2, hidden_size * 2)
        self.out_head = classification_head()
        self.dropout = th.nn.Dropout(dropout_rate)
        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

    def forward(self, text, attention_mask, sample_ids):
        bool_mask = attention_mask.to(th.bool)

        tokens_emb = self.embedding(text) * attention_mask[:, :, None]
        input_states = self.convert_layer(tokens_emb)

        tokens_emb, _ = self.encoder(tokens_emb)
        tokens_emb = self.layer_norm(tokens_emb + input_states)
        tokens_emb = self.dropout(tokens_emb)

        attention_weights = self.attention_fc(tokens_emb)
        attention_weights = attention_weights.masked_fill_(
            ~bool_mask[:, :, None], th.finfo(th.float).min
        )

        if self.training:
            attention_noises = th.normal(
                0,
                self.noise_sigma,
                attention_weights.size(),
                device=attention_weights.device,
                dtype=attention_weights.dtype,
            )
            attention_noises = th.abs(attention_noises).masked_fill_(
                ~bool_mask[:, :, None], th.finfo(th.float).min
            )
            attention_weights += attention_noises

        attention_weights = th.nn.functional.softmax(attention_weights, dim=1)
        tokens_emb = self.projection_fc(attention_weights * tokens_emb)
        tokens_emb = self.activation(tokens_emb)
        tokens_emb = self.layer_norm(tokens_emb)

        final_states = self.pooling(tokens_emb, attention_mask)
        final_states = self.dropout(final_states)
        final_logits = self.out_head(final_states)

        return attention_weights, final_logits
