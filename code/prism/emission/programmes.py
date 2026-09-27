"""Programme emission.

Ref: Methods Sec. 6.1 -- the model output is not a risk score but a programme pi = h(Z): a
subset of the metabolite vocabulary with a sign for each entry, together with a predictive
effect for every supported programme and an interval around that effect. Ref: Algorithm 1
steps 15-16 -- the supported set comes from the support of the decoder and the signs from its
coefficients, while the effects and intervals come from the calibrated interval head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from prism.cohort.lexicon import Lexicon
from prism.errors import ConfigError
from prism.modelling.interval_head import z_multiplier
from prism.modelling.programme_head import (
    SupportThreshold,
    emissions_from_coefficients,
    select_support,
    sign_agreement,
    signs_for_support,
    support_sizes,
)


@dataclass(frozen=True)
class ProgrammeStatement:
    """The reported output for one observation."""

    record_id: str
    supported_metabolites: tuple[str, ...]
    signs: Mapping[str, int]
    supported_programmes: tuple[str, ...]
    effects: Mapping[str, float]
    intervals: Mapping[str, tuple[float, float]]

    def interval_width(self, programme_id: str) -> float:
        low, high = self.intervals[programme_id]
        return high - low

    def effect_over_width(self, programme_id: str) -> float:
        """Effect against its own interval width, the reading a user applies to a programme."""
        width = self.interval_width(programme_id)
        if width <= 0.0:
            raise ConfigError("an effect cannot be sized against a zero-width interval")
        return abs(self.effects[programme_id]) / width

    def as_dict(self) -> Mapping[str, object]:
        return {
            "record_id": self.record_id,
            "supported_metabolites": list(self.supported_metabolites),
            "signs": {name: int(value) for name, value in sorted(self.signs.items())},
            "supported_programmes": list(self.supported_programmes),
            "effects": {name: float(value) for name, value in sorted(self.effects.items())},
            "intervals": {
                name: [float(low), float(high)]
                for name, (low, high) in sorted(self.intervals.items())
            },
        }


def _coefficient_tensor(coefficients: np.ndarray) -> Tensor:
    values = np.asarray(coefficients, dtype=np.float32)
    if values.ndim != 2:
        raise ConfigError("decoder coefficients are read as rows by vocabulary entries")
    return torch.from_numpy(values)


def emit_programmes(
    coefficients: np.ndarray,
    record_ids: Sequence[str],
    lexicon: Lexicon,
    beta: np.ndarray,
    scale: np.ndarray,
    *,
    nominal_coverage: float,
    threshold: SupportThreshold | None = None,
) -> tuple[ProgrammeStatement, ...]:
    """Turn decoder coefficients and interval-head output into reported programmes."""
    tensor = _coefficient_tensor(coefficients)
    if int(tensor.shape[1]) != lexicon.size:
        raise ConfigError("decoder coefficients must be (rows, vocabulary_size)")
    n_rows = int(tensor.shape[0])
    if len(record_ids) != n_rows:
        raise ConfigError("one record identifier is needed per coefficient row")
    effects = np.asarray(beta, dtype=np.float64)
    spreads = np.asarray(scale, dtype=np.float64)
    if effects.shape != (n_rows, lexicon.n_blocks()) or spreads.shape != effects.shape:
        raise ConfigError("the interval head must emit one effect and scale per programme")
    if not 0.0 < nominal_coverage < 1.0:
        raise ConfigError("a nominal coverage must lie strictly inside (0, 1)")

    multiplier = z_multiplier(nominal_coverage)
    base = emissions_from_coefficients(tensor, lexicon, tuple(record_ids), threshold)
    statements: list[ProgrammeStatement] = []
    for row in range(n_rows):
        entry = base[row]
        statement_effects: dict[str, float] = {}
        statement_intervals: dict[str, tuple[float, float]] = {}
        for programme_id in entry.programme_support:
            position = lexicon.programme_ids.index(programme_id)
            centre = float(effects[row, position])
            half = float(spreads[row, position]) * multiplier
            statement_effects[programme_id] = centre
            statement_intervals[programme_id] = (centre - half, centre + half)
        statements.append(
            ProgrammeStatement(
                record_id=entry.record_id,
                supported_metabolites=entry.supported,
                signs=entry.signs,
                supported_programmes=entry.programme_support,
                effects=statement_effects,
                intervals=statement_intervals,
            )
        )
    return tuple(statements)


def statement_table(
    statements: Sequence[ProgrammeStatement],
) -> tuple[Mapping[str, object], ...]:
    return tuple(statement.as_dict() for statement in statements)


def support_summary(statements: Sequence[ProgrammeStatement]) -> Mapping[str, float]:
    """Average number of supported metabolites and programmes per observation."""
    if not statements:
        raise ConfigError("a support summary needs at least one statement")
    metabolites = float(np.mean([len(s.supported_metabolites) for s in statements]))
    programmes = float(np.mean([len(s.supported_programmes) for s in statements]))
    return {"mean_supported_metabolites": metabolites, "mean_supported_programmes": programmes}


def effect_table(
    statements: Sequence[ProgrammeStatement], programme_ids: Sequence[str]
) -> Mapping[str, Mapping[str, float]]:
    """Per-programme effect and interval width across a cohort."""
    table: dict[str, dict[str, float]] = {}
    for programme_id in programme_ids:
        effects = [
            statement.effects[programme_id]
            for statement in statements
            if programme_id in statement.effects
        ]
        widths = [
            statement.interval_width(programme_id)
            for statement in statements
            if programme_id in statement.intervals
        ]
        if not effects:
            continue
        table[programme_id] = {
            "mean_effect": float(np.mean(effects)),
            "sd_effect": float(np.std(effects, ddof=0)),
            "mean_interval_width": float(np.mean(widths)),
            "n_observations": float(len(effects)),
        }
    return table


def programme_ranking(statements: Sequence[ProgrammeStatement]) -> tuple[tuple[str, float], ...]:
    """Programmes ordered by mean effect over the cohort.

    Ref: Results Sec. 2.1 -- the emitted programme is reported as a named program with an
    estimated effect, so the ranking is a reported quantity rather than a diagnostic.
    """
    if not statements:
        raise ConfigError("a ranking needs at least one statement")
    totals: dict[str, list[float]] = {}
    for statement in statements:
        for programme_id, effect in statement.effects.items():
            totals.setdefault(programme_id, []).append(effect)
    ranked = [(name, float(np.mean(values))) for name, values in totals.items()]
    return tuple(sorted(ranked, key=lambda item: (-abs(item[1]), item[0])))


def sign_agreement_of_statements(
    statements: Sequence[ProgrammeStatement], lexicon: Lexicon
) -> float:
    """Share of emitted signs that match the reported direction of movement."""
    expected = {entry.name: entry.expected_sign for entry in lexicon.catalogue()}
    agree = 0
    total = 0
    for statement in statements:
        for name, sign in statement.signs.items():
            reference = expected.get(name)
            if reference is None:
                continue
            total += 1
            if int(sign) == reference:
                agree += 1
    if total == 0:
        return 0.0
    return agree / float(total)


def restricted_to(
    statement: ProgrammeStatement, lexicon: Lexicon, programmes: Sequence[str]
) -> ProgrammeStatement:
    """The part of a statement that belongs to a named set of programmes."""
    keep = {
        lexicon.name_of(index)
        for programme_id in programmes
        for index in lexicon.block_members(programme_id)
    }
    return replace(
        statement,
        supported_metabolites=tuple(
            name for name in statement.supported_metabolites if name in keep
        ),
        signs={name: sign for name, sign in statement.signs.items() if name in keep},
        supported_programmes=tuple(
            programme for programme in statement.supported_programmes if programme in programmes
        ),
        effects={
            programme: value
            for programme, value in statement.effects.items()
            if programme in programmes
        },
        intervals={
            programme: value
            for programme, value in statement.intervals.items()
            if programme in programmes
        },
    )


def emission_diagnostics(
    coefficients: np.ndarray, lexicon: Lexicon, expected_signs: Mapping[str, int]
) -> Mapping[str, float]:
    """Sparsity, support sizes and sign agreement for one coefficient block."""
    tensor = _coefficient_tensor(coefficients)
    if int(tensor.shape[1]) != lexicon.size:
        raise ConfigError("decoder coefficients must be (rows, vocabulary_size)")
    sizes = support_sizes(select_support(tensor, lexicon), lexicon)
    diagnostics = {f"support_{key}": float(value) for key, value in sizes.items()}
    diagnostics["sign_agreement"] = sign_agreement(tensor, lexicon, expected_signs)
    return diagnostics


def signs_at_threshold(
    coefficients: np.ndarray, lexicon: Lexicon, threshold: SupportThreshold
) -> tuple[tuple[int, ...], ...]:
    """Signs of the supported entries at an explicit threshold, in index space."""
    tensor = _coefficient_tensor(coefficients)
    support = select_support(tensor, lexicon, threshold)
    return signs_for_support(tensor, support)
