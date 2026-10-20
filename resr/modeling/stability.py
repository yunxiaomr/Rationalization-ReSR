import torch as th
import torch.nn.functional as F


def target_relative_rate_bounds(
    target_rate: float,
    *,
    lower_factor: float,
    upper_factor: float,
    upper_cap: float = 0.95,
) -> tuple[float, float]:

    target_rate = float(target_rate)
    lower_factor = float(lower_factor)
    upper_factor = float(upper_factor)
    upper_cap = float(upper_cap)
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be in (0, 1), got {target_rate}")
    if not 0.0 < lower_factor <= 1.0:
        raise ValueError(f"lower_factor must be in (0, 1], got {lower_factor}")
    if upper_factor < 1.0:
        raise ValueError(f"upper_factor must be at least 1, got {upper_factor}")
    if not 0.0 < upper_cap <= 1.0:
        raise ValueError(f"upper_cap must be in (0, 1], got {upper_cap}")

    lower = target_rate * lower_factor
    upper = min(upper_cap, target_rate * upper_factor)
    if lower >= upper:
        raise ValueError(f"Invalid target-relative interval: [{lower}, {upper}]")
    return lower, upper


def structural_checkpoint_loss(
    *,
    validation_loss: th.Tensor,
    validation_rate: th.Tensor,
    target_rate: float,
    lower_factor: float = 0.5,
    upper_factor: float = 2.0,
    infeasible_base: float = 1000.0,
) -> tuple[th.Tensor, th.Tensor]:

    lower, upper = target_relative_rate_bounds(
        target_rate,
        lower_factor=lower_factor,
        upper_factor=upper_factor,
    )
    safe_loss = th.nan_to_num(
        validation_loss.to(dtype=th.float32),
        nan=float(infeasible_base),
        posinf=float(infeasible_base),
        neginf=0.0,
    )
    rate = th.nan_to_num(
        validation_rate.to(device=safe_loss.device, dtype=safe_loss.dtype),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    lower_tensor = th.tensor(lower, device=rate.device, dtype=rate.dtype)
    upper_tensor = th.tensor(upper, device=rate.device, dtype=rate.dtype)
    violation = th.relu(lower_tensor - rate) + th.relu(rate - upper_tensor)
    feasible = violation <= 1e-8
    normalized_violation = violation / max(float(target_rate), 1e-6)
    infeasible_loss = (
        float(infeasible_base)
        + normalized_violation
        + 1e-3 * safe_loss.clamp(min=0.0, max=float(infeasible_base))
    )
    return th.where(feasible, safe_loss, infeasible_loss), feasible.to(dtype=th.float32)


def deterministic_budget_mask(
    *,
    highlight_logits: th.Tensor,
    attention_mask: th.Tensor,
    stochastic_mask: th.Tensor,
    temperature: float,
    enabled: bool,
    mask_padding: bool = True,
) -> tuple[th.Tensor, th.Tensor]:

    temperature = max(float(temperature), 1e-6)
    selection_prob = F.softmax(highlight_logits / temperature, dim=-1)[..., 1]
    deterministic_mask = th.argmax(highlight_logits, dim=-1).to(dtype=selection_prob.dtype)
    constraint_mask = deterministic_mask + selection_prob - selection_prob.detach()
    if not enabled:
        constraint_mask = stochastic_mask

    if mask_padding:
        valid_mask = attention_mask.to(dtype=selection_prob.dtype)
        constraint_mask = constraint_mask * valid_mask
        selection_prob = selection_prob * valid_mask
    return constraint_mask, selection_prob


def mask_rates(mask: th.Tensor, attention_mask: th.Tensor) -> th.Tensor:
    valid_mask = attention_mask.to(dtype=mask.dtype)
    return (mask * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1.0)


def collapse_rate_recovery_loss(
    selection_prob: th.Tensor,
    attention_mask: th.Tensor,
    target_rate: float,
) -> th.Tensor:

    soft_rates = mask_rates(selection_prob, attention_mask)
    target_rates = th.full_like(soft_rates, float(target_rate))
    epsilon = 1e-6
    return (
        -target_rates * th.log(soft_rates + epsilon)
        - (1.0 - target_rates) * th.log(1.0 - soft_rates + epsilon)
    ).mean()
