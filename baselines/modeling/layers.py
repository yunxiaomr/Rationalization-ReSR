import torch as th


class SelectItem(th.nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = "selectitem"
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class AttentionPooling(th.nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        super().__init__()
        self.attention = th.nn.Sequential(
            th.nn.Linear(in_dim, hidden_size),
            th.nn.LayerNorm(hidden_size),
            th.nn.GELU(),
            th.nn.Linear(hidden_size, 1),
        )
        self._weights = None

    @property
    def attn_weights(self):
        return self._weights

    def forward(self, hidden_states: th.Tensor, masks: th.Tensor = None):
        weight = self.attention(hidden_states).squeeze()
        if masks is not None:
            masks = masks.bool()
            weight.masked_fill_(~masks, th.finfo(th.float).min)
        weight = th.nn.functional.softmax(weight, dim=-1)
        self._weights = weight
        weight = weight.unsqueeze(dim=1)
        converge_representations = th.bmm(weight, hidden_states).squeeze()
        return converge_representations


class FactorAnnealer:
    def __init__(self, factor: float, decay_callback):
        self.factor = factor
        self.decay_callback = decay_callback
        self.current_step = 0
        self._current_factor = factor

    def step(self):
        decay_factor = self.decay_callback(self.current_step)
        self.current_step += 1
        self._current_factor = self.factor * decay_factor
        return self._current_factor

    @property
    def current_factor(self):
        return self._current_factor
