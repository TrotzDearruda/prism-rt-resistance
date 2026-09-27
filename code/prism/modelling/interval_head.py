"""Interval head.

Ref: Methods Sec. 6.6 -- the expected effect of a programme and the area around it are
produced by a dedicated head whose nominal reliability target is fixed before fitting;
the coverage is exact under the no-shift conditions and is then measured again under
shift. Ref: Algorithm 1 step 16 -- the head is validated leave-one-site-out. Ref: Table 8
-- the reported configuration is read as a mean interval width next to an empirical
coverage.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist

import torch
from torch import Tensor, nn

from prism.errors import ConfigError


@dataclass(frozen=True)
class IntervalReport:
    """Width and coverage of a set of intervals, read against a nominal target."""

    nominal_coverage: float
    empirical_coverage: float
    mean_width: float
    n_intervals: int

    def coverage_gap(self) -> float:
        return self.empirical_coverage - self.nominal_coverage

    def covers_nominal(self, tolerance: float) -> bool:
        return abs(self.coverage_gap()) <= tolerance


class IntervalHead(nn.Module):
    """Per-programme point effect and scale."""

    net: nn.Sequential
    out: nn.Linear
    scale_floor: float

    def __init__(
        self,
        in_dim: int,
        n_programmes: int,
        hidden: int,
        *,
        scale_floor: float = 1e-3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim < 1 or n_programmes < 1 or hidden < 1:
            raise ConfigError("an interval head needs positive widths")
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden, 2 * n_programmes)
        self.scale_floor = scale_floor
        self.n_programmes = n_programmes

    def forward(self, representation: Tensor) -> tuple[Tensor, Tensor]:
        """Return the point effect and a positive scale for every programme."""
        hidden: Tensor = self.net(representation)
        raw: Tensor = self.out(hidden).reshape(-1, self.n_programmes, 2)
        beta: Tensor = raw[:, :, 0]
        scale: Tensor = torch.nn.functional.softplus(raw[:, :, 1]) + self.scale_floor
        return beta, scale


def z_multiplier(coverage: float) -> float:
    """Two-sided multiplier for a nominal coverage."""
    if not 0.0 < coverage < 1.0:
        raise ConfigError("a nominal coverage must lie strictly inside (0, 1)")
    return float(NormalDist().inv_cdf(1.0 - 0.5 * (1.0 - coverage)))


def interval_from_scale(beta: Tensor, scale: Tensor, coverage: float) -> tuple[Tensor, Tensor]:
    """Symmetric interval around the point effect."""
    if beta.shape != scale.shape:
        raise ConfigError("the point effect and its scale must share a shape")
    multiplier = z_multiplier(coverage)
    lower: Tensor = beta - multiplier * scale
    upper: Tensor = beta + multiplier * scale
    return lower, upper


def interval_width(lower: Tensor, upper: Tensor) -> Tensor:
    if lower.shape != upper.shape:
        raise ConfigError("interval bounds must share a shape")
    width: Tensor = upper - lower
    return width


def report_intervals(
    lower: Tensor,
    upper: Tensor,
    targets: Tensor,
    nominal_coverage: float,
) -> IntervalReport:
    """Coverage and width of a batch of intervals against the held-out effects."""
    if lower.shape != upper.shape or lower.shape != targets.shape:
        raise ConfigError("intervals and targets must share a shape")
    width = interval_width(lower, upper)
    inside: Tensor = ((targets >= lower) & (targets <= upper)).to(targets.dtype)
    return IntervalReport(
        nominal_coverage=nominal_coverage,
        empirical_coverage=float(inside.mean()),
        mean_width=float(width.mean()),
        n_intervals=int(targets.numel()),
    )


def leave_one_site_out(
    lower_by_site: Mapping[str, Tensor],
    upper_by_site: Mapping[str, Tensor],
    target_by_site: Mapping[str, Tensor],
    nominal_coverage: float,
) -> IntervalReport:
    """Pool the per-site folds into one coverage estimate.

    Ref: Algorithm 1 step 16 -- the interval head is validated on the site that was left
    out, so the pooled read-out is over sites rather than over rows.
    """
    if set(lower_by_site) != set(upper_by_site) or set(lower_by_site) != set(target_by_site):
        raise ConfigError("every fold must carry bounds and targets for the same site")
    if not lower_by_site:
        raise ConfigError("no fold was supplied")
    lowers = torch.cat([lower_by_site[site].reshape(-1) for site in sorted(lower_by_site)])
    uppers = torch.cat([upper_by_site[site].reshape(-1) for site in sorted(upper_by_site)])
    targets = torch.cat([target_by_site[site].reshape(-1) for site in sorted(target_by_site)])
    return report_intervals(lowers, uppers, targets, nominal_coverage)


def coverage_shift(reference: IntervalReport, shifted: IntervalReport) -> Mapping[str, float]:
    """Degradation of coverage and width when the setting shifts.

    Ref: Methods Sec. 6.6 -- coverage is measured again under shift, and Ref: Table 8 --
    a wider interval that keeps its coverage is reported differently from a narrow one
    that loses it.
    """
    return {
        "coverage_reference": reference.empirical_coverage,
        "coverage_shifted": shifted.empirical_coverage,
        "coverage_delta": shifted.empirical_coverage - reference.empirical_coverage,
        "width_reference": reference.mean_width,
        "width_shifted": shifted.mean_width,
        "width_delta": shifted.mean_width - reference.mean_width,
    }
