# Prism -- constrained multimodal adaptation for rectal chemoradiotherapy response

This repository is the code release that accompanies *A Multimodal Foundation Model Charting
Metabolic Reprogramming Underlying Radiotherapy Resistance in Colorectal Cancer Across Globally
Diverse Care Settings*. It implements the model the paper calls Prism: a multimodal backbone whose
representation is fitted against a spatial-metabolic alignment term, a cross-setting invariance
penalty and a sparse signed programme decoder, under a discrimination floor that the article declares
before any model is fitted.

The output of the model is not a risk score. It is a programme: a subset of a metabolite vocabulary
with a sign for each entry, together with a per-programme effect and an interval around that effect.

Two things about this release are worth knowing before reading further.

* The clinical arm the study used is held by the participating institutions and is not
  redistributable. This repository ships the reader for that arm, not the arm and not a way to
  manufacture one: a run reads the archive the data owner staged and says plainly when it is absent.
  The code paths are exercised by the verification harness on fixture records, and no value measured
  on them is a study result. The public resources the article cites are read from their own services.
* The article prints no hyper-parameters -- its own algorithm caption says that no experiment numeric
  value appears in it. Every schedule value here is therefore an engineering default, declared in
  `configs/` and recorded per field in each run's header, together with the values the article *does*
  fix (the equivalence margin, the null band, the family size, the coverage target).

## Layout

```
code/prism/notation.py         the symbols of the constrained problem: X, Y, S, M, Z, g, h
code/prism/cohort/             records, care settings, partitions, prevalence, supervision, readers
code/prism/modelling/          per-modality encoders, fusion, modality dropout, heads, assembly
code/prism/objectives/         Eq. (1) term by term, and the feasibility constraint of Eq. (2)
code/prism/fitting/            optimiser, schedule, checkpoints, the mini-batch engine, the stages
code/prism/measurement/        discrimination, calibration, decisions, attribution, transfer
code/prism/statistics/         paired comparison, multiplicity control, equivalence, the fixed family
code/prism/emission/           the reported programme, its effects and its intervals
code/prism/perturbation/       the in-vitro assay design and its pre-declared bands
code/prism/assessment/         the analysis plan, the ablation matrix, the table assembly
code/prism/reference/          the values transcribed from the article and the resources it cites
code/prism/audit/              the claim map, release hygiene and verification bookkeeping
code/prism/cli/                train, evaluate, emit, report, audit
code/checks/                   the independent verification harness
code/tests/                    the test suite
configs/                       data, model, train and experiment configuration
scripts/                       launch and data-staging helpers
```

## Installation

Python 3.11 or newer, with a CUDA build of PyTorch for the reported configuration.

```bash
pip install -e ".[dev]"
```

```bash
conda env create -f environment.yml
conda activate prism
```

```bash
docker build -t prism:0.4 .
```

## Data

The nine public resources the auxiliary comparison names, with the sample counts the article prints
beside them, are listed in `dataset_urls.txt` together with the route each was checked by and the date
of that check. Accessions and counts were read back from the source services rather than transcribed
from the manuscript. Six of the nine are annotated radiotherapy-response series; the three open-tier
genomic and proteogenomic resources carry no response annotation and are used for representation
pretraining and world knowledge only.

```bash
bash scripts/prepare_data.sh
```

The script stages the series it can reach under `data/` and prints a per-resource ledger. A resource
that cannot be reached from your network is reported and skipped; the annotated series are tens of
megabytes each, and their metadata block -- which carries every sample identifier and is what the
sample count is checked against -- sits at the front of the file.

### Staging the arm

The clinical arm is not obtainable from this repository and there is no code path that manufactures
one. A run reads a single archive: a compressed `.npz` holding one row per record with

| group | contents |
| --- | --- |
| identity | `record_id`, `patient_id`, `site`, `region`, `stratum`, `outcome`, `layer` |
| streams | `stream__<modality>` stacked per stream, `shape__<modality>`, `availability` |
| supervision | `metabolite`, `supervised` |

Stacked means a record whose site could not acquire a modality carries zeros in that stream and the
availability matrix marks the row, so the modality-dropout scheme can read what the site could
actually supply. `modality_names`, `schema_version` and `provenance` are written alongside, and the
reader refuses an archive whose streams do not match the configuration it is being read with.

Stage it at `data/cohort.npz`, or pass `--cohort PATH`. A run without one reports that the arm is
absent rather than substituting anything for it.

## Training

One command per reported configuration; `--experiment` names the file under `configs/experiment/`.

```bash
python -m prism.cli train --experiment main --cohort data/cohort.npz --device cuda
python -m prism.cli train --experiment comparator_unconstrained --cohort data/cohort.npz --device cuda
python -m prism.cli train --experiment comparator_adversarial --cohort data/cohort.npz --device cuda
python -m prism.cli train --experiment ablation_minus_invariance --cohort data/cohort.npz --device cuda
```

The primary configuration is a ten-member seed ensemble of 32 records per step, an accumulation
window of two and a world size of eight, so an effective batch of 512. Each member fits five probe
epochs with the encoders held fixed and up to thirty constrained epochs, early-stopping on a
held-out block of the development arm that is cut at the patient level.

A distributed launch uses the same entry point:

```bash
bash scripts/launch_train.sh main 8
```

## Evaluation

```bash
python -m prism.cli evaluate --experiment main --cohort data/cohort.npz --device cuda
python -m prism.cli emit --experiment main --partition external --cohort data/cohort.npz
python -m prism.cli report --experiment main
```

`evaluate` writes the predictor comparison, the representation-level attribution panel and, where a
region carries enough rows on both sides of a transfer, the cross-region matrix. `emit` writes the
reported programme, its per-programme effects and intervals, its support size and its sign agreement
with the reported directions of movement. `report` re-reads the tables a run wrote, re-applies the
interval footer, and records where a measured value stands against the transcribed one.

Because the clinical arm is unavailable, the values of the article's own tables cannot be produced
here and are not claimed. What a run writes is the measured value plus the interval footer, and the
comparison against the article is written out cell by cell with the gap and whether it falls inside a
stated tolerance. Where the arm is absent the tables are not written at all.

## Verification

```bash
PYTHONPATH=code python3 code/checks/run_all.py
```

The driver runs two layers and writes `claim_to_code.json`, `verification_report.json` and
`integrity_manifest.json`.

* The **manuscript-arithmetic** layer recomputes the numbers the article states and the relations it
  asserts between them, from a second transcription of its tables and an independently written set of
  routines. Its purpose is to separate a contradiction inside the article from a defect in this code:
  the two are scored apart, so a failed arithmetic check is not read as a broken release.
* The **code-behaviour** layer executes the release on fixture records built by the harness and on
  the real public services: batch tensorisation, a forward and backward pass through the objective of
  Eq. (1), an optimiser step, a single-batch overfit, a checkpoint round trip, the two-stage smoke
  loop, the modality-dropout scheme, the archive round trip, the emission path, the interval coverage
  read-out, the paired comparison and the step-down correction against independent implementations,
  the conditional reading, the assay guards, the ablation matrix, the attribution probe, the scaling
  reductions, the resource list, and the command line end to end. The fixture builder lives in
  `code/checks/fixtures.py` and is imported by nothing in `code/prism/`.

Every claim ends in one of four states. `PASS` and `FAIL` mean it was executed and held or did not.
`NOT_RUN` means no execution applies -- a value the article never fixes cannot be checked against one.
`BLOCKED` means the data it needs are not obtainable here, so nothing was executed and nothing is
claimed. A release whose claims are split between these states is reported as `PARTIALLY_VERIFIED`.

The same pass also checks the release itself: no machine-specific path, no generated-content phrase,
no credential-shaped string, no module that nothing imports, no two files with the same content, and
a manifest that lists every shipped file and not its own report.

## Compute

The article reports no hardware, no wall clock and no storage. The numbers below are what this
release's own configuration implies and what it was executed at, and they are not presented as the
study's.

| Quantity | Primary configuration | Executed here |
| --- | --- | --- |
| Accelerator | one 8-GPU node, mixed precision | CPU only |
| Memory per device | 40 GB or more for the volumes named in `configs/model/backbone.yaml` | -- |
| Effective batch | 512 | 16 |
| Ensemble members | 10 | 1 |
| Storage | the six annotated series, tens of megabytes each, plus checkpoints | a few megabytes |

The full primary configuration is not runnable on a CPU alone; the smoke configuration in
`configs/experiment/_smoke.yaml` is, and it is what the test suite and the smoke probes use.

## Tests

```bash
PYTHONPATH=code python3 -m pytest
ruff check .
python3 -m mypy --strict code/prism code/checks
```

## What the article does not fix

Recorded here so that a reader can tell a reported value from a default. The claim map in
`claim_to_code.json` carries the same list with the location each field was read from.

* Every schedule value, every width and depth, the vocabulary size and its block contents, and the
  three objective weights are engineering defaults.
* The cohort assembly, endpoints and provenance have a heading and no text in the article; the
  partition structure here is reconstructed from the results section and from the record counts that
  are printed.
* The site-to-region assignment, the metabolite vocabulary and two of the six programme names are
  reconstructions, marked as such.
* The decoder section is cross-referenced by a section number the article does not contain, and the
  ablation caption's exemption from the null band is not the only one the table supports. Both are
  reported by the arithmetic layer rather than reconciled quietly.

## Licence

Released under the MIT licence; see `LICENSE`. Third-party material and its terms are recorded in
`THIRD_PARTY_NOTICES.txt`.
