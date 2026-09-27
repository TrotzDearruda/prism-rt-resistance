"""Conditional value of the programme once the oxygenation proxy is entered.

Ref: Results Sec. 2.1 and Discussion Sec. 4.1 -- the test of whether the programme is an axis of
its own or a re-description of hypoxia is run by entering the oxygenation proxy as an additional
covariate and asking what each programme still adds. Three of the six retain incremental value,
the hypoxia-linked one loses all of it, and the rest are too small or inconsistent to be
conclusive. The reading is reported programme by programme rather than as one test, because the
manuscript treats it as several tests at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from prism.cohort.pathways import (
    INCONCLUSIVE,
    LOSES,
    RETAINS,
    conditional_verdicts,
    retained_programmes,
    unconstrained_substitution_reading,
)
from prism.errors import ConfigError
from prism.measurement.calibration import fit_logistic
from prism.measurement.discrimination import rank_auc


def _with_intercept(columns: Sequence[np.ndarray]) -> np.ndarray:
    blocks = [np.asarray(column, dtype=np.float64).reshape(-1) for column in columns]
    if not blocks:
        raise ConfigError("a covariate design needs at least one column")
    width = blocks[0].size
    for block in blocks:
        if block.size != width:
            raise ConfigError("covariate columns must share a length")
    return np.column_stack([np.ones(width, dtype=np.float64), *blocks])


def fitted_auc(design: np.ndarray, labels: np.ndarray) -> float:
    """Discrimination of a logistic fit on a covariate design.

    The statistic is read on the fitted linear predictor, which is monotone in the fitted
    probability, so the transform between the two does not enter the value.
    """
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    if design.shape[0] != truth.size:
        raise ConfigError("a covariate design needs one row per label")
    if np.unique(truth > 0.5).size < 2:
        raise ConfigError("a conditional reading needs both classes present")
    coefficients = fit_logistic(design, truth)
    linear = design @ coefficients
    return rank_auc(linear, truth)


def conditional_gain(
    outcome_logit: np.ndarray,
    oxygenation_proxy: np.ndarray,
    programme_effect: np.ndarray,
    labels: np.ndarray,
    *,
    fit_fraction: float = 0.6,
    seed: int = 20240921,
) -> float:
    """Incremental value of one programme on top of the outcome and the proxy.

    The comparison is between the same fit with and without the programme column, so the number is
    the contribution of that programme in the presence of a proxy for oxygenation rather than the
    contribution of the programme by itself. The fit is taken on one part of the rows and the
    discrimination is read on the rest: a gain measured in sample would be non-negative for any
    column at all, including a column of noise, so a programme that adds nothing could never be
    reported as adding nothing.
    """
    if not 0.0 < fit_fraction < 1.0:
        raise ConfigError("a conditional reading needs a fit fraction inside (0, 1)")
    base = _with_intercept(
        [
            np.asarray(outcome_logit, dtype=np.float64),
            np.asarray(oxygenation_proxy, dtype=np.float64),
        ]
    )
    augmented = _with_intercept(
        [
            np.asarray(outcome_logit, dtype=np.float64),
            np.asarray(oxygenation_proxy, dtype=np.float64),
            np.asarray(programme_effect, dtype=np.float64),
        ]
    )
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    if base.shape[0] != truth.size:
        raise ConfigError("a conditional reading needs one label per row")
    rng = np.random.default_rng(seed)
    order = rng.permutation(truth.size)
    cut = max(10, min(truth.size - 10, int(round(fit_fraction * truth.size))))
    fit_rows = order[:cut]
    held_rows = order[cut:]
    if np.unique(truth[fit_rows] > 0.5).size < 2 or np.unique(truth[held_rows] > 0.5).size < 2:
        raise ConfigError("both halves of a conditional reading need both classes present")
    base_coefficients = fit_logistic(base[fit_rows], truth[fit_rows])
    augmented_coefficients = fit_logistic(augmented[fit_rows], truth[fit_rows])
    base_score: np.ndarray = base[held_rows] @ base_coefficients
    augmented_score: np.ndarray = augmented[held_rows] @ augmented_coefficients
    return rank_auc(augmented_score, truth[held_rows]) - rank_auc(base_score, truth[held_rows])


@dataclass(frozen=True)
class ConditionalRow:
    """One programme's conditional reading."""

    programme_id: str
    gain: float
    verdict: str

    def as_dict(self) -> Mapping[str, object]:
        return {
            "programme": self.programme_id,
            "incremental_gain": self.gain,
            "verdict": self.verdict,
        }


def conditional_table(gains: Mapping[str, float], null_band: float) -> tuple[ConditionalRow, ...]:
    """Turn measured gains into the three-way verdict per programme."""
    verdicts = conditional_verdicts(gains, null_band)
    return tuple(
        ConditionalRow(programme_id=key, gain=float(value), verdict=verdicts[key])
        for key, value in gains.items()
    )


def reading_summary(rows: Sequence[ConditionalRow], null_band: float) -> Mapping[str, object]:
    """What the conditional test establishes, in the manuscript's own three readings.

    Ref: Results Sec. 2.1 -- the programme is at least partly self-sufficient with respect to
    oxygenation, the hypoxia-linked block is the one that is not, and the direction of the
    insignificant residual is reported as the genuine outcome of a test that could have produced
    the opposite result.
    """
    if not rows:
        raise ConfigError("a conditional summary needs at least one row")
    verdicts = {row.programme_id: row.verdict for row in rows}
    retained = retained_programmes()
    held = [programme_id for programme_id in retained if verdicts.get(programme_id) == RETAINS]
    lost = [key for key, value in verdicts.items() if value == LOSES]
    thin = [key for key, value in verdicts.items() if value == INCONCLUSIVE]
    return {
        "n_programmes": float(len(rows)),
        "n_retained": float(len(held)),
        "n_lost": float(len(lost)),
        "n_inconclusive": float(len(thin)),
        "retained_minus_lost": float(len(held) - len(lost)),
        "null_band": float(null_band),
        "substitution_reading": unconstrained_substitution_reading(verdicts),
    }


def proxy_shift(base_external: float, with_proxy_external: float) -> Mapping[str, float]:
    """Movement of external discrimination when the proxy is entered.

    A movement smaller than the null band is reported as no movement, in either direction, rather
    than as a small improvement or a small loss.
    """
    delta = float(with_proxy_external) - float(base_external)
    return {"external_delta": delta, "sign": np.sign(delta)}


def programme_effect_columns(
    effects: Sequence[Mapping[str, float]], programme_ids: Sequence[str]
) -> Mapping[str, np.ndarray]:
    """Per-programme effect columns from emitted statement effects, in a fixed programme order.

    A programme absent from a statement contributes a zero column entry rather than being dropped,
    so every column keeps one row per observation and the designs stay aligned.
    """
    if not effects:
        raise ConfigError("a conditional reading needs at least one statement")
    columns: dict[str, np.ndarray] = {}
    for programme_id in programme_ids:
        values = [float(statement.get(programme_id, 0.0)) for statement in effects]
        columns[programme_id] = np.asarray(values, dtype=np.float64)
    return columns
