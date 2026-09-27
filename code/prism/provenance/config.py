"""Typed configuration intake.

The manuscript states that all hyper-parameters are declared before any model is fitted
(Algorithm 1), so configuration is read once, validated structurally, and then frozen
into dataclasses that the rest of the package sees. Values that the manuscript does not
report carry their origin in ``ORIGIN`` below and are overridden per experiment file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import yaml

from prism.errors import ConfigError

ORIGIN_ENGINEERING_DEFAULT = "engineering-default"
ORIGIN_MANUSCRIPT = "manuscript"

INVARIANCE_MODES: tuple[str, ...] = ("variance", "adversarial")


@dataclass(frozen=True)
class StreamConfig:
    """One modality encoder."""

    name: str
    input_shape: tuple[int, ...]
    width: int
    depth: int
    dropout: float = 0.0
    origin: str = ORIGIN_ENGINEERING_DEFAULT


@dataclass(frozen=True)
class ModelConfig:
    """Backbone geometry, fusion and the heads attached to the fused representation."""

    streams: Mapping[str, StreamConfig]
    fused_dim: int
    representation_dim: int
    probe_hidden: int
    interval_hidden: int
    decoder_hidden: int
    site_classifier_hidden: int
    origin: str = ORIGIN_ENGINEERING_DEFAULT


@dataclass(frozen=True)
class VocabularyConfig:
    """Metabolite vocabulary and its metabolically coherent blocks.

    Ref: Methods Sec. 6.6 -- sparsity is imposed by a group penalty over metabolically
    coherent blocks rather than element-wise.
    """

    size: int
    blocks: tuple[tuple[int, ...], ...]
    programme_ids: tuple[str, ...]
    origin: str = ORIGIN_ENGINEERING_DEFAULT

    def n_blocks(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class CohortConfig:
    """Cohort sizes, care-setting structure and the partition boundary.

    Ref: Results Sec. 2.1 (3,847 held out, 1,024 silent) and Sec. 2.6 (sites A-E develop,
    F and G hold out).
    """

    n_records: int
    n_silent: int
    prevalence: float
    prevalence_ci: tuple[float, float]
    sites: tuple[str, ...]
    development_sites: tuple[str, ...]
    holdout_sites: tuple[str, ...]
    regions: tuple[str, ...]
    origin: str = ORIGIN_MANUSCRIPT


@dataclass(frozen=True)
class SupervisionConfig:
    """How much of the retrospective arm carries spatially resolved metabolites."""

    fraction: float
    block_coherence: float
    noise: float
    origin: str = ORIGIN_ENGINEERING_DEFAULT


@dataclass(frozen=True)
class ObjectiveConfig:
    """Weights and shape of the terms of Eq. (1).

    Ref: Methods Sec. 6.1 Eq. (1); Sec. 6.4 (alignment); Sec. 6.5 (invariance);
    Sec. 6.6 (programme). The three weights are declared before fitting and are not
    quoted in the manuscript, so the values here are engineering defaults.
    """

    lambda_alignment: float = 1.0
    lambda_invariance: float = 0.5
    lambda_programme: float = 0.1
    lambda_sign: float = 0.05
    covariance_weight: float = 1.0
    matching_temperature: float = 0.1
    invariance_mode: str = "variance"
    group_penalty: float = 1.0
    origin: str = ORIGIN_ENGINEERING_DEFAULT


@dataclass(frozen=True)
class TrainConfig:
    """Two-stage schedule, optimiser settings and the pre-declared discrimination floor.

    Ref: Algorithm 1 (Stage 1 freezes the encoders, Stage 2 releases them under the
    feasibility constraint of Eq. (2)); Methods Sec. 6.9 (equivalence margin); Sec. 6.10
    (null band and multiplicity family).
    """

    stage1_epochs: int
    stage2_epochs: int
    batch_size: int
    grad_accum: int
    world_size: int
    lr_head: float
    lr_encoder: float
    weight_decay: float
    warmup_fraction: float
    grad_clip: float
    precision: str
    patience: int
    seed: int
    ensemble_members: int
    modality_dropout: float
    discrimination_floor: float
    equivalence_margin: float
    null_band: float
    multiplicity_family: int
    interval_nominal_coverage: float
    origin: str = ORIGIN_ENGINEERING_DEFAULT


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything one run needs, assembled from the four config groups."""

    name: str
    arm: str
    cohort: CohortConfig
    supervision: SupervisionConfig
    model: ModelConfig
    objective: ObjectiveConfig
    train: TrainConfig
    vocabulary: VocabularyConfig
    notes: tuple[str, ...] = field(default_factory=tuple)

    def effective_batch(self) -> int:
        return self.train.batch_size * self.train.grad_accum * self.train.world_size

    def with_train(self, **changes: Any) -> ExperimentConfig:
        return replace(self, train=replace(self.train, **changes))

    def with_objective(self, **changes: Any) -> ExperimentConfig:
        return replace(self, objective=replace(self.objective, **changes))


def stream_shapes(model: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Per-stream shapes as the configuration declares them.

    A staged arm is read at these shapes, so a model built from a configuration can read the arm a
    run is configured for. Without this the two would have to agree by coincidence, and a
    configuration describing a 256-dimensional patch block would meet a 32-dimensional one.
    """
    return {name: stream.input_shape for name, stream in model.streams.items()}


def _require(mapping: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where} is missing required key {key!r}")
    return mapping[key]


def _as_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a mapping, found {type(value).__name__}")
    return cast(Mapping[str, Any], value)


def _as_sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ConfigError(f"{where} must be a sequence, found {type(value).__name__}")
    return value


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, found {type(value).__name__}")
    return float(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number, found {type(value).__name__}")
    return int(value)


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string, found {type(value).__name__}")
    return value


def _as_shape(value: Any, where: str) -> tuple[int, ...]:
    items = _as_sequence(value, where)
    return tuple(_as_int(item, f"{where}[]") for item in items)


def load_yaml(path: Path) -> Mapping[str, Any]:
    """Read one YAML document and return it as a mapping."""
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    if parsed is None:
        raise ConfigError(f"config file is empty: {path}")
    return _as_mapping(parsed, str(path))


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge; ``overlay`` wins on scalars and merges on mappings."""
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(
                cast(Mapping[str, Any], current), cast(Mapping[str, Any], value)
            )
        else:
            merged[key] = value
    return merged


def apply_overrides(root: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``a.b.c=value`` overrides onto a config tree."""
    tree: dict[str, Any] = dict(root)
    for expression in overrides:
        if "=" not in expression:
            raise ConfigError(f"override {expression!r} is not of the form key=value")
        dotted, raw = expression.split("=", 1)
        parts = [part for part in dotted.split(".") if part]
        if not parts:
            raise ConfigError(f"override {expression!r} carries no key")
        cursor: dict[str, Any] = tree
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, Mapping):
                nxt = {}
                cursor[part] = nxt
            cursor = cast(dict[str, Any], nxt)
        cursor[parts[-1]] = yaml.safe_load(raw)
    return tree


def _stream_config(name: str, raw: Mapping[str, Any]) -> StreamConfig:
    return StreamConfig(
        name=name,
        input_shape=_as_shape(
            _require(raw, "input_shape", f"model.streams.{name}"),
            f"model.streams.{name}.input_shape",
        ),
        width=_as_int(
            _require(raw, "width", f"model.streams.{name}"), f"model.streams.{name}.width"
        ),
        depth=_as_int(
            _require(raw, "depth", f"model.streams.{name}"), f"model.streams.{name}.depth"
        ),
        dropout=_as_float(raw.get("dropout", 0.0), f"model.streams.{name}.dropout"),
    )


def parse_config(raw: Mapping[str, Any], name: str) -> ExperimentConfig:
    """Validate a merged config tree and freeze it into dataclasses."""
    cohort_raw = _as_mapping(_require(raw, "cohort", name), f"{name}.cohort")
    supervision_raw = _as_mapping(_require(raw, "supervision", name), f"{name}.supervision")
    vocab_raw = _as_mapping(_require(raw, "vocabulary", name), f"{name}.vocabulary")
    model_raw = _as_mapping(_require(raw, "model", name), f"{name}.model")
    objective_raw = _as_mapping(_require(raw, "objective", name), f"{name}.objective")
    train_raw = _as_mapping(_require(raw, "train", name), f"{name}.train")

    sites = tuple(
        _as_str(item, f"{name}.cohort.sites[]")
        for item in _as_sequence(_require(cohort_raw, "sites", "cohort"), "cohort.sites")
    )
    development = tuple(
        _as_str(item, f"{name}.cohort.development_sites[]")
        for item in _as_sequence(
            _require(cohort_raw, "development_sites", "cohort"), "cohort.development_sites"
        )
    )
    holdout = tuple(
        _as_str(item, f"{name}.cohort.holdout_sites[]")
        for item in _as_sequence(
            _require(cohort_raw, "holdout_sites", "cohort"), "cohort.holdout_sites"
        )
    )
    regions = tuple(
        _as_str(item, f"{name}.cohort.regions[]")
        for item in _as_sequence(_require(cohort_raw, "regions", "cohort"), "cohort.regions")
    )
    prevalence_ci_raw = _as_sequence(
        _require(cohort_raw, "prevalence_ci", "cohort"), "cohort.prevalence_ci"
    )
    if len(prevalence_ci_raw) != 2:
        raise ConfigError("cohort.prevalence_ci needs exactly two entries")

    cohort = CohortConfig(
        n_records=_as_int(_require(cohort_raw, "n_records", "cohort"), "cohort.n_records"),
        n_silent=_as_int(_require(cohort_raw, "n_silent", "cohort"), "cohort.n_silent"),
        prevalence=_as_float(_require(cohort_raw, "prevalence", "cohort"), "cohort.prevalence"),
        prevalence_ci=(
            _as_float(prevalence_ci_raw[0], "cohort.prevalence_ci[0]"),
            _as_float(prevalence_ci_raw[1], "cohort.prevalence_ci[1]"),
        ),
        sites=sites,
        development_sites=development,
        holdout_sites=holdout,
        regions=regions,
        origin=_as_str(cohort_raw.get("origin", ORIGIN_MANUSCRIPT), "cohort.origin"),
    )
    if not set(development) <= set(sites) or not set(holdout) <= set(sites):
        raise ConfigError("development and holdout sites must be drawn from cohort.sites")
    if set(development) & set(holdout):
        raise ConfigError("a site cannot both develop the model and hold out from it")

    supervision = SupervisionConfig(
        fraction=_as_float(
            _require(supervision_raw, "fraction", "supervision"), "supervision.fraction"
        ),
        block_coherence=_as_float(
            _require(supervision_raw, "block_coherence", "supervision"),
            "supervision.block_coherence",
        ),
        noise=_as_float(_require(supervision_raw, "noise", "supervision"), "supervision.noise"),
    )

    blocks_raw = _as_sequence(_require(vocab_raw, "blocks", "vocabulary"), "vocabulary.blocks")
    blocks: list[tuple[int, ...]] = []
    for index, block in enumerate(blocks_raw):
        members = _as_sequence(block, f"vocabulary.blocks[{index}]")
        blocks.append(tuple(_as_int(item, f"vocabulary.blocks[{index}][]") for item in members))
    vocabulary = VocabularyConfig(
        size=_as_int(_require(vocab_raw, "size", "vocabulary"), "vocabulary.size"),
        blocks=tuple(blocks),
        programme_ids=tuple(
            _as_str(item, "vocabulary.programme_ids[]")
            for item in _as_sequence(
                _require(vocab_raw, "programme_ids", "vocabulary"), "vocabulary.programme_ids"
            )
        ),
    )
    covered = sorted({index for block in vocabulary.blocks for index in block})
    if covered != list(range(vocabulary.size)):
        raise ConfigError("vocabulary blocks must partition the metabolite vocabulary")
    if len(vocabulary.programme_ids) != len(vocabulary.blocks):
        raise ConfigError("each programme needs one metabolically coherent block")

    streams_raw = _as_mapping(_require(model_raw, "streams", "model"), "model.streams")
    streams = {
        _as_str(key, "model.streams key"): _stream_config(
            str(key), _as_mapping(value, f"model.streams.{key}")
        )
        for key, value in streams_raw.items()
    }
    if not streams:
        raise ConfigError("model.streams must name at least one modality")
    model = ModelConfig(
        streams=streams,
        fused_dim=_as_int(_require(model_raw, "fused_dim", "model"), "model.fused_dim"),
        representation_dim=_as_int(
            _require(model_raw, "representation_dim", "model"), "model.representation_dim"
        ),
        probe_hidden=_as_int(_require(model_raw, "probe_hidden", "model"), "model.probe_hidden"),
        interval_hidden=_as_int(
            _require(model_raw, "interval_hidden", "model"), "model.interval_hidden"
        ),
        decoder_hidden=_as_int(
            _require(model_raw, "decoder_hidden", "model"), "model.decoder_hidden"
        ),
        site_classifier_hidden=_as_int(
            _require(model_raw, "site_classifier_hidden", "model"),
            "model.site_classifier_hidden",
        ),
    )

    mode = _as_str(objective_raw.get("invariance_mode", "variance"), "objective.invariance_mode")
    if mode not in INVARIANCE_MODES:
        raise ConfigError(f"objective.invariance_mode must be one of {INVARIANCE_MODES}")
    objective = ObjectiveConfig(
        lambda_alignment=_as_float(
            objective_raw.get("lambda_alignment", 1.0), "objective.lambda_alignment"
        ),
        lambda_invariance=_as_float(
            objective_raw.get("lambda_invariance", 0.5), "objective.lambda_invariance"
        ),
        lambda_programme=_as_float(
            objective_raw.get("lambda_programme", 0.1), "objective.lambda_programme"
        ),
        lambda_sign=_as_float(objective_raw.get("lambda_sign", 0.05), "objective.lambda_sign"),
        covariance_weight=_as_float(
            objective_raw.get("covariance_weight", 1.0), "objective.covariance_weight"
        ),
        matching_temperature=_as_float(
            objective_raw.get("matching_temperature", 0.1),
            "objective.matching_temperature",
        ),
        invariance_mode=mode,
        group_penalty=_as_float(objective_raw.get("group_penalty", 1.0), "objective.group_penalty"),
    )

    train = TrainConfig(
        stage1_epochs=_as_int(_require(train_raw, "stage1_epochs", "train"), "train.stage1_epochs"),
        stage2_epochs=_as_int(_require(train_raw, "stage2_epochs", "train"), "train.stage2_epochs"),
        batch_size=_as_int(_require(train_raw, "batch_size", "train"), "train.batch_size"),
        grad_accum=_as_int(_require(train_raw, "grad_accum", "train"), "train.grad_accum"),
        world_size=_as_int(_require(train_raw, "world_size", "train"), "train.world_size"),
        lr_head=_as_float(_require(train_raw, "lr_head", "train"), "train.lr_head"),
        lr_encoder=_as_float(_require(train_raw, "lr_encoder", "train"), "train.lr_encoder"),
        weight_decay=_as_float(_require(train_raw, "weight_decay", "train"), "train.weight_decay"),
        warmup_fraction=_as_float(
            _require(train_raw, "warmup_fraction", "train"), "train.warmup_fraction"
        ),
        grad_clip=_as_float(_require(train_raw, "grad_clip", "train"), "train.grad_clip"),
        precision=_as_str(_require(train_raw, "precision", "train"), "train.precision"),
        patience=_as_int(_require(train_raw, "patience", "train"), "train.patience"),
        seed=_as_int(_require(train_raw, "seed", "train"), "train.seed"),
        ensemble_members=_as_int(
            _require(train_raw, "ensemble_members", "train"), "train.ensemble_members"
        ),
        modality_dropout=_as_float(
            _require(train_raw, "modality_dropout", "train"), "train.modality_dropout"
        ),
        discrimination_floor=_as_float(
            _require(train_raw, "discrimination_floor", "train"),
            "train.discrimination_floor",
        ),
        equivalence_margin=_as_float(
            _require(train_raw, "equivalence_margin", "train"), "train.equivalence_margin"
        ),
        null_band=_as_float(_require(train_raw, "null_band", "train"), "train.null_band"),
        multiplicity_family=_as_int(
            _require(train_raw, "multiplicity_family", "train"), "train.multiplicity_family"
        ),
        interval_nominal_coverage=_as_float(
            _require(train_raw, "interval_nominal_coverage", "train"),
            "train.interval_nominal_coverage",
        ),
    )
    if train.batch_size < 1 or train.grad_accum < 1 or train.world_size < 1:
        raise ConfigError("batch_size, grad_accum and world_size must be positive")
    if not 0.0 < train.modality_dropout < 1.0:
        raise ConfigError("train.modality_dropout must lie strictly inside (0, 1)")

    notes = tuple(
        _as_str(item, f"{name}.notes[]")
        for item in _as_sequence(raw.get("notes", []), f"{name}.notes")
    )
    arm = _as_str(raw.get("arm", "constrained"), f"{name}.arm")
    return ExperimentConfig(
        name=name,
        arm=arm,
        cohort=cohort,
        supervision=supervision,
        model=model,
        objective=objective,
        train=train,
        vocabulary=vocabulary,
        notes=notes,
    )


GROUP_DIRECTORIES: Mapping[str, str] = {
    "cohort": "data",
    "supervision": "data",
    "vocabulary": "data",
    "model": "model",
    "objective": "train",
    "train": "train",
}


def load_experiment(path: Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load one experiment file, merging the shared groups it names.

    A group given as a bare name is read from the directory that owns it, so an experiment file
    carries only the values that make it different from its siblings.
    """
    raw = load_yaml(path)
    merged: dict[str, Any] = {}
    for group, values in raw.items():
        if group in {"include", "name", "arm", "notes"}:
            continue
        if isinstance(values, str):
            directory = GROUP_DIRECTORIES.get(str(group))
            if directory is None:
                raise ConfigError(f"the group {group!r} has no directory")
            merged[group] = load_yaml(path.parent.parent / directory / f"{values}.yaml")
        else:
            merged[group] = values
    merged["name"] = raw.get("name", path.stem)
    merged["arm"] = raw.get("arm", "constrained")
    merged["notes"] = raw.get("notes", [])
    merged = deep_merge(merged, {})
    for override in overrides:
        merged = apply_overrides(merged, [override])
    return parse_config(merged, str(merged["name"]))
