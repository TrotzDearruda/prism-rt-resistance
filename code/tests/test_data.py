"""Tensorisation of records and the modality dropout scheme."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.cohort.schema import Layer
from prism.cohort.tensors import (
    batch_availability,
    build_batch,
    describe_batch,
    labels_for,
    metabolite_for,
    move_batch,
    region_index_for,
    site_index_for,
    stack_bag,
)
from prism.errors import SupervisionError
from prism.modelling.modality_dropout import (
    any_subset_estimable,
    availability_tensor,
    coverage_histogram,
    draw_dropout_plan,
)
from prism.notation import MODALITY_STREAMS


def test_a_batch_carries_the_shapes_its_encoders_read(development) -> None:
    rows = np.arange(16, dtype=np.int64)
    batch = build_batch(development, rows)
    assert batch.size() == 16
    assert batch.present.shape == (16, len(MODALITY_STREAMS))
    for name, tensor in batch.streams.items():
        assert tensor.shape[0] == 16, name
    description = describe_batch(batch)
    assert description["rows"] == 16
    assert batch.record_ids[0] == development.records[0].record_id


def test_a_site_that_cannot_acquire_a_modality_supplies_absolutely_nothing(development) -> None:
    weakest = development.by_stratum("lower-middle-resource")
    if len(weakest) == 0:
        pytest.skip("the fixture records hold no low-resource record")
    batch = build_batch(weakest, np.arange(min(8, len(weakest)), dtype=np.int64))
    assert "mri" not in batch.streams
    assert "histology" not in batch.streams
    assert batch.present[:, 1].sum() == 0


def test_labels_and_index_vectors_line_up(development) -> None:
    rows = np.arange(12, dtype=np.int64)
    labels = labels_for(development, rows)
    sites = site_index_for(development, rows)
    regions = region_index_for(development, rows)
    assert labels.shape == sites.shape == regions.shape == (12,)
    assert int(sites.max()) < len(development.sites())
    assert int(regions.max()) < len(development.regions())
    assert set(labels.tolist()) <= {0.0, 1.0}


def test_a_bag_stream_is_padded_and_masked() -> None:
    class Record:
        def __init__(self, name, block, availability):
            self.record_id = name
            self.features = {"histology": block}
            self.availability = availability

        def has(self, modality):
            return modality in self.availability

    records = [
        Record("a", np.ones((3, 4), dtype=np.float32), ("histology",)),
        Record("b", np.full((5, 4), 2.0, dtype=np.float32), ("histology",)),
        Record("c", np.zeros((1, 4), dtype=np.float32), ()),
    ]
    padded, mask = stack_bag(records, "histology")
    assert padded.shape == (3, 5, 4)
    assert mask.sum().item() == 3 + 5
    assert not bool(mask[2].any())


def test_dropout_never_enables_a_stream_a_site_lacks(development) -> None:
    rows = np.arange(24, dtype=np.int64)
    availability = torch.from_numpy(batch_availability(development, rows)).to(torch.bool)
    generator = torch.Generator().manual_seed(3)
    plan = draw_dropout_plan(availability, 0.5, generator)
    assert bool((~plan.present | availability).all())
    assert any_subset_estimable(plan)
    histogram = coverage_histogram(plan)
    assert sum(histogram.values()) == 24


def test_dropout_rescues_a_row_that_would_be_left_empty(development) -> None:
    availability = torch.ones((4, len(MODALITY_STREAMS)), dtype=torch.bool)
    generator = torch.Generator().manual_seed(11)
    plan = draw_dropout_plan(availability, 0.99, generator)
    assert plan.n_rescued > 0
    assert bool((plan.present.sum(dim=1) >= 1).all())


def test_availability_tensor_reflects_the_record_lists(development) -> None:
    rows = np.arange(6, dtype=np.int64)
    tensor = availability_tensor([development.records[int(row)].availability for row in rows])
    assert tensor.shape == (6, len(MODALITY_STREAMS))
    assert float(tensor.sum()) > 0.0


def test_metabolite_blocks_come_from_supervised_rows(development) -> None:
    supervised = [
        position for position, record in enumerate(development.records) if record.supervised()
    ]
    if not supervised:
        pytest.skip("no supervised record among the fixture records")
    block = metabolite_for(development, np.asarray(supervised[:5], dtype=np.int64))
    assert block.shape[0] == min(5, len(supervised))
    with pytest.raises(SupervisionError):
        metabolite_for(
            development,
            np.asarray(
                [
                    position
                    for position, record in enumerate(development.records)
                    if not record.supervised()
                ][:3],
                dtype=np.int64,
            ),
        )


def test_moving_a_batch_keeps_its_identity(development) -> None:
    batch = build_batch(development, np.arange(4, dtype=np.int64))
    moved = move_batch(batch, torch.device("cpu"))
    assert moved.record_ids == batch.record_ids
    assert moved.present.shape == batch.present.shape


def test_the_silent_layer_is_never_used_for_fitting(parts) -> None:
    fitted = parts[Layer.DEVELOPMENT]
    held = parts.get(Layer.EXTERNAL)
    assert held is not None
    assert set(fitted.patient_ids()).isdisjoint(set(held.patient_ids()))
