from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VRSRHyperparams:
    scheduler_policy: str = "vrsr"
    r_star: float = 0.15
    epsilon: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    eta: float = 0.0
    mu: float = 1.0
    nu: float = 0.0
    q: float = 1.0
    warmup_epochs: int = 0
    inherit_gap_threshold: float = 0.0
    inherit_patience: int = 0
    inherit_slices: int = 2
    probe_gap_threshold: float = 0.0
    predictor_gap_threshold: float = 0.0
    gap_ema_decay: float = 0.0
    collapse_min_rate: float = 0.0
    collapse_max_rate: float = 1.0
    recovery_slices: int = 2
    use_generator_scheduler_guidance: bool = True
    use_predictor_scheduler_guidance: bool = True


@dataclass
class VRSRState:
    vr_g: float = 0.0
    vr_p: float = 0.0
    age_g: int = 0
    age_p: int = 0
    w_g: float = 1.0
    w_p: float = 1.0
    next_phase: str = "predictor"
    prev_val_rate: Optional[float] = None
    warmup_remaining: int = 0
    bottleneck_g_count: int = 0
    bottleneck_p_count: int = 0
    force_phase: Optional[str] = None
    force_remaining: int = 0
    ema_generator_gap: float = 0.0
    ema_predictor_gap: float = 0.0
    gap_ema_seen: int = 0
    collapse_active: bool = False
    collapse_events: int = 0
    last_reason: str = "init"


@dataclass
class VRSRDecision:
    next_phase: str
    d_g: float
    d_p: float
    generator_gap: float
    predictor_gap: float
    raw_generator_gap: float
    raw_predictor_gap: float
    w_g: float
    w_p: float
    reason: str
    collapse_active: bool
    collapse_events: int


def _safe_float(x) -> float:
    try:
        return float(x)
    except TypeError:
        return 0.0


def _generator_loss_gap(
    l_probe: float,
    l_ref: float,
    epsilon: float,
) -> float:
    return max(0.0, _safe_float(l_probe) - (_safe_float(l_ref) + _safe_float(epsilon)))


def _predictor_loss_gap(
    l_online: float,
    l_probe: float,
) -> float:
    return max(0.0, _safe_float(l_online) - _safe_float(l_probe))


def _threshold_gap(value: float, threshold: float) -> float:
    return max(0.0, _safe_float(value) - max(0.0, _safe_float(threshold)))


def _ema_gap(
    *,
    value: float,
    previous: float,
    seen: int,
    decay: float,
) -> float:
    decay = min(max(_safe_float(decay), 0.0), 0.999)
    value = max(0.0, _safe_float(value))
    if decay <= 0.0 or seen <= 0:
        return value
    return decay * _safe_float(previous) + (1.0 - decay) * value


def compute_generator_perf_gap(
    *,
    l_probe: float,
    l_ref: float,
    epsilon: float,
    hp: VRSRHyperparams,
) -> float:
    return _generator_loss_gap(l_probe=l_probe, l_ref=l_ref, epsilon=epsilon)


def compute_predictor_lag_gap(
    *,
    l_online: float,
    l_probe: float,
    hp: VRSRHyperparams,
) -> float:
    return _predictor_loss_gap(l_online=l_online, l_probe=l_probe)


def compute_generator_debt(
    *,
    l_probe: float,
    l_ref: float,
    epsilon: float,
    rate: float,
    r_star: float,
    frag: float,
    drift: float,
    age_g: int,
    hp: VRSRHyperparams,
    perf_gap_override: Optional[float] = None,
) -> float:
    perf_gap = (
        _safe_float(perf_gap_override)
        if perf_gap_override is not None
        else compute_generator_perf_gap(
            l_probe=l_probe,
            l_ref=l_ref,
            epsilon=epsilon,
            hp=hp,
        )
    )
    return (
        hp.alpha * perf_gap
        + hp.beta * abs(_safe_float(rate) - _safe_float(r_star))
        + hp.gamma * _safe_float(frag)
        + hp.delta * _safe_float(drift)
        + hp.eta * float(age_g)
    )


def compute_predictor_debt(
    *,
    l_online: float,
    l_probe: float,
    age_p: int,
    hp: VRSRHyperparams,
    lag_gap_override: Optional[float] = None,
) -> float:
    lag_gap = (
        _safe_float(lag_gap_override)
        if lag_gap_override is not None
        else compute_predictor_lag_gap(
            l_online=l_online,
            l_probe=l_probe,
            hp=hp,
        )
    )
    return hp.mu * lag_gap + hp.nu * float(age_p)


class VRSRScheduler:
    def __init__(
        self,
        hp: VRSRHyperparams,
        state: Optional[VRSRState] = None,
    ):
        self.hp = hp
        if self.hp.scheduler_policy not in {"vrsr", "fixed_alternating", "loss_gap"}:
            raise ValueError(f"Unsupported scheduler policy: {self.hp.scheduler_policy}")
        self.state = state if state is not None else VRSRState()
        if self.state.warmup_remaining <= 0 and self.hp.warmup_epochs > 0:
            self.state.warmup_remaining = int(self.hp.warmup_epochs)

    def on_slice_end(
        self,
        executed_phase: str,
    ):
        if self.state.warmup_remaining > 0:
            self.state.warmup_remaining -= 1

        if self.state.force_remaining > 0 and self.state.force_phase == executed_phase:
            self.state.force_remaining -= 1
            if self.state.force_remaining <= 0:
                self.state.force_phase = None
                self.state.force_remaining = 0

        if executed_phase == "generator":
            self.state.vr_g += self.hp.q / max(self.state.w_g, 1e-8)
            self.state.age_g = 0
            self.state.age_p += 1
        else:
            self.state.vr_p += self.hp.q / max(self.state.w_p, 1e-8)
            self.state.age_p = 0
            self.state.age_g += 1

    def decide(
        self,
        *,
        l_online: float,
        l_probe: float,
        l_ref: float,
        l_probe_predictor: Optional[float] = None,
        val_rate: float,
        val_frag: float,
        val_drift: float,
    ) -> VRSRDecision:
        l_probe_for_predictor = (
            _safe_float(l_probe_predictor)
            if l_probe_predictor is not None
            else _safe_float(l_probe)
        )

        raw_generator_gap = _generator_loss_gap(
            l_probe=l_probe,
            l_ref=l_ref,
            epsilon=self.hp.epsilon,
        )
        raw_predictor_gap = _predictor_loss_gap(
            l_online=l_online,
            l_probe=l_probe_for_predictor,
        )
        generator_gap = _threshold_gap(raw_generator_gap, self.hp.probe_gap_threshold)
        predictor_gap = _threshold_gap(raw_predictor_gap, self.hp.predictor_gap_threshold)
        if self.hp.gap_ema_decay > 0.0:
            self.state.ema_generator_gap = _ema_gap(
                value=generator_gap,
                previous=self.state.ema_generator_gap,
                seen=self.state.gap_ema_seen,
                decay=self.hp.gap_ema_decay,
            )
            self.state.ema_predictor_gap = _ema_gap(
                value=predictor_gap,
                previous=self.state.ema_predictor_gap,
                seen=self.state.gap_ema_seen,
                decay=self.hp.gap_ema_decay,
            )
            self.state.gap_ema_seen += 1
            generator_gap = self.state.ema_generator_gap
            predictor_gap = self.state.ema_predictor_gap

        generator_perf_gap = generator_gap

        if self.hp.scheduler_policy in {"fixed_alternating", "loss_gap"}:
            d_g = generator_gap if self.hp.scheduler_policy == "loss_gap" else 0.0
        elif self.hp.use_generator_scheduler_guidance:
            d_g = compute_generator_debt(
                l_probe=l_probe,
                l_ref=l_ref,
                epsilon=self.hp.epsilon,
                rate=val_rate,
                r_star=self.hp.r_star,
                frag=val_frag,
                drift=val_drift,
                age_g=self.state.age_g,
                hp=self.hp,
                perf_gap_override=generator_perf_gap,
            )
        else:
            d_g = 0.0

        if self.hp.scheduler_policy in {"fixed_alternating", "loss_gap"}:
            d_p = predictor_gap if self.hp.scheduler_policy == "loss_gap" else 0.0
        elif self.hp.use_predictor_scheduler_guidance:
            d_p = compute_predictor_debt(
                l_online=l_online,
                l_probe=l_probe_for_predictor,
                age_p=self.state.age_p,
                hp=self.hp,
                lag_gap_override=predictor_gap,
            )
        else:
            d_p = 0.0

        if self.hp.scheduler_policy in {"fixed_alternating", "loss_gap"}:
            w_g = 1.0
            w_p = 1.0
        else:
            w_g = 1.0 + max(0.0, _safe_float(d_g))
            w_p = 1.0 + max(0.0, _safe_float(d_p))

        bottleneck_p = predictor_gap if self.hp.use_predictor_scheduler_guidance else 0.0
        if self.hp.inherit_patience > 0 and bottleneck_p > self.hp.inherit_gap_threshold:
            self.state.bottleneck_p_count += 1
        else:
            self.state.bottleneck_p_count = 0

        bottleneck_g = generator_perf_gap if self.hp.use_generator_scheduler_guidance else 0.0
        if self.hp.inherit_patience > 0 and bottleneck_g > self.hp.inherit_gap_threshold:
            self.state.bottleneck_g_count += 1
        else:
            self.state.bottleneck_g_count = 0

        reason = "vrsr"

        collapsed = (
            _safe_float(val_rate) < self.hp.collapse_min_rate
            or _safe_float(val_rate) > self.hp.collapse_max_rate
        )
        if collapsed:
            if not self.state.collapse_active:
                self.state.collapse_events += 1
            self.state.collapse_active = True
            self.state.force_phase = "generator"
            self.state.force_remaining = int(self.hp.recovery_slices)
            reason = "mask_collapse"
        elif self.state.collapse_active and self.state.force_remaining <= 0:
            self.state.collapse_active = False

        if self.state.warmup_remaining > 0:
            next_phase = "predictor"
            reason = "warmup"
        elif collapsed:
            next_phase = "generator"
            reason = "mask_collapse"
        elif self.state.force_remaining > 0 and self.state.force_phase is not None:
            next_phase = self.state.force_phase
            reason = f"forced:{self.state.force_phase}"
        elif (
            self.hp.inherit_patience > 0
            and self.state.bottleneck_p_count >= self.hp.inherit_patience
        ):
            self.state.force_phase = "predictor"
            self.state.force_remaining = int(self.hp.inherit_slices)
            self.state.bottleneck_p_count = 0
            next_phase = "predictor"
            reason = "inherit:predictor"
        elif (
            self.hp.inherit_patience > 0
            and self.state.bottleneck_g_count >= self.hp.inherit_patience
        ):
            self.state.force_phase = "generator"
            self.state.force_remaining = int(self.hp.inherit_slices)
            self.state.bottleneck_g_count = 0
            next_phase = "generator"
            reason = "inherit:generator"
        elif self.hp.scheduler_policy == "loss_gap":
            if generator_gap > predictor_gap:
                next_phase = "generator"
            elif predictor_gap > generator_gap:
                next_phase = "predictor"
            else:
                next_phase = "generator" if self.state.vr_g < self.state.vr_p else "predictor"
            reason = "loss_gap"
        else:
            score_g = self.state.vr_g / max(w_g, 1e-8)
            score_p = self.state.vr_p / max(w_p, 1e-8)
            next_phase = "generator" if score_g < score_p else "predictor"
            if self.hp.scheduler_policy == "fixed_alternating":
                reason = "fixed_alternating"

        self.state.w_g = w_g
        self.state.w_p = w_p
        self.state.next_phase = next_phase
        self.state.prev_val_rate = _safe_float(val_rate)
        self.state.last_reason = reason

        return VRSRDecision(
            next_phase=next_phase,
            d_g=_safe_float(d_g),
            d_p=_safe_float(d_p),
            generator_gap=_safe_float(generator_gap),
            predictor_gap=_safe_float(predictor_gap),
            raw_generator_gap=_safe_float(raw_generator_gap),
            raw_predictor_gap=_safe_float(raw_predictor_gap),
            w_g=_safe_float(w_g),
            w_p=_safe_float(w_p),
            reason=reason,
            collapse_active=bool(self.state.collapse_active),
            collapse_events=int(self.state.collapse_events),
        )
