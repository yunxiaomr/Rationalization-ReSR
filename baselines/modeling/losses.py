import torch as th


class SparsityConstraint(th.nn.Module):
    def __init__(
        self,
        sparsity_level,
        per_example: bool = False,
        mask_padding: bool = False,
    ):
        super().__init__()

        self.sparsity_level = sparsity_level
        self.per_example = bool(per_example)
        self.mask_padding = bool(mask_padding)

    def forward(
        self,
        highlight_hat,
        attention_mask,
    ):

        attention_mask = attention_mask.to(dtype=highlight_hat.dtype)
        selected = highlight_hat * attention_mask if self.mask_padding else highlight_hat

        if self.per_example:
            valid_tokens = attention_mask.sum(dim=-1)
            valid_examples = valid_tokens > 0
            rates = selected.sum(dim=-1) / valid_tokens.clamp_min(1.0)
            if bool(valid_examples.any()):
                return th.abs(rates[valid_examples] - self.sparsity_level).mean()
            return selected.sum() * 0.0

        sparsity = selected.sum() / attention_mask.sum().clamp_min(1.0)
        return th.abs(sparsity - self.sparsity_level)


class ContinuityConstraint(th.nn.Module):
    def forward(
        self,
        highlight_mask,
        attention_mask=None,
    ):

        if highlight_mask.shape[1] <= 1:
            return highlight_mask.sum() * 0.0

        differences = th.abs(highlight_mask[:, 1:] - highlight_mask[:, :-1])
        if attention_mask is None:
            return differences.mean()

        attention_mask = attention_mask.to(dtype=highlight_mask.dtype)
        valid_adjacencies = attention_mask[:, 1:] * attention_mask[:, :-1]
        return (differences * valid_adjacencies).sum() / valid_adjacencies.sum().clamp_min(1.0)


class JS_Div(th.nn.Module):
    def __init__(self):
        super(JS_Div, self).__init__()
        self.kl_div = th.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p, q):
        p_s = th.nn.functional.softmax(p, dim=-1)
        q_s = th.nn.functional.softmax(q, dim=-1)
        p_s, q_s = p_s.view(-1, p_s.size(-1)), q_s.view(-1, q_s.size(-1))
        m = (0.5 * (p_s + q_s)).log()
        return 0.5 * (self.kl_div(m, p_s.log()) + self.kl_div(m, q_s.log()))


class KL_Div(th.nn.Module):
    def forward(self, p, q):
        return th.nn.functional.kl_div(
            th.nn.functional.softmax(p, dim=-1).log(),
            th.nn.functional.softmax(q, dim=-1),
            reduction="batchmean",
        )
