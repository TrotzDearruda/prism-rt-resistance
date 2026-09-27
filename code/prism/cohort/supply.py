"""Readers for the public resources of Table 5.

The annotated series are read from the GEO series matrix text format, whose header carries the
sample-level annotation before the expression table begins. The open-tier genomic resources are read
from the GDC REST service. Both readers keep what they transfer bounded: a series matrix is tens of
megabytes compressed while its metadata sits at the front, so a sample-count check reads only the
header block, and the GDC endpoint returns project metadata only.
"""

from __future__ import annotations

import gzip
import io
import json
import shutil
import subprocess
import time
import urllib.request
import zlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prism.errors import ResourceError
from prism.provenance.digest import digest_bytes
from prism.provenance.journal import get_logger
from prism.reference.urls import DatasetResource, resource

USER_AGENT = "prism-release/0.4"
NETWORK_TIMEOUT_SECONDS = 45
CONNECT_TIMEOUT_SECONDS = 10
CLIENT_TIMEOUT_SECONDS = 120
NETWORK_ATTEMPTS = 4
READ_CHUNK_BYTES = 1 << 20
HEADER_BUDGET_BYTES = 4 << 20
MAX_SERIES_MATRIX_BYTES = 96 * 1024 * 1024

ENDPOINT_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "pcr": ("pcr", "pathologic complete", "complete response", "complete pathologic"),
    "response": ("response", "responder", "regression", "trg", "tumour regression"),
    "sensitivity": ("sensitiv", "sensitive", "radiosensitiv"),
    "resistance": ("resist", "non-responder", "nonresponder", "poor response"),
}


@dataclass(frozen=True)
class GeoSeries:
    """A parsed series matrix, or the readable part of one."""

    accession: str
    sample_ids: tuple[str, ...]
    titles: tuple[str, ...]
    characteristic_fields: tuple[str, ...]
    characteristics: tuple[Mapping[str, str], ...]
    feature_ids: tuple[str, ...]
    matrix: np.ndarray
    source_url: str
    sha256: str
    partial: bool = False

    def n_samples(self) -> int:
        return len(self.sample_ids)

    def n_features(self) -> int:
        return len(self.feature_ids)

    def field_values(self, field_name: str) -> tuple[str, ...]:
        return tuple(str(entry.get(field_name, "")) for entry in self.characteristics)

    def labels_for(self, keywords: Iterable[str]) -> tuple[int, ...]:
        """Score each sample against a keyword set, using the annotation text alone."""
        needles = tuple(keyword.lower() for keyword in keywords)
        labels: list[int] = []
        for entry in self.characteristics:
            text = " ".join(f"{key} {value}" for key, value in entry.items()).lower()
            labels.append(1 if any(needle in text for needle in needles) else 0)
        return tuple(labels)


@dataclass(frozen=True)
class GeoSeriesBundle:
    """All platform matrices of one series, read as one annotated cohort."""

    accession: str
    platforms: tuple[GeoSeries, ...]

    def n_samples(self) -> int:
        return sum(series.n_samples() for series in self.platforms)

    def sample_ids(self) -> tuple[str, ...]:
        collected: list[str] = []
        for series in self.platforms:
            collected.extend(series.sample_ids)
        return tuple(collected)

    def characteristics(self) -> tuple[Mapping[str, str], ...]:
        collected: list[Mapping[str, str]] = []
        for series in self.platforms:
            collected.extend(series.characteristics)
        return tuple(collected)

    def titles(self) -> tuple[str, ...]:
        collected: list[str] = []
        for series in self.platforms:
            collected.extend(series.titles)
        return tuple(collected)

    def n_features(self) -> int:
        return sum(series.n_features() for series in self.platforms)

    def platform_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for series in self.platforms:
            token = series.source_url.rsplit("/", 1)[-1]
            platform = token.split("-", 1)[1].split("_", 1)[0] if "-" in token else "GPL?"
            names.append(platform)
        return tuple(names)

    def digests(self) -> tuple[str, ...]:
        return tuple(series.sha256 for series in self.platforms)

    def partial(self) -> bool:
        return any(series.partial for series in self.platforms)

    def labels_for(self, keywords: Iterable[str]) -> tuple[int, ...]:
        needles = tuple(keyword.lower() for keyword in keywords)
        labels: list[int] = []
        for entry in self.characteristics():
            text = " ".join(f"{key} {value}" for key, value in entry.items()).lower()
            labels.append(1 if any(needle in text for needle in needles) else 0)
        return tuple(labels)

    def field_values(self, field_name: str) -> tuple[str, ...]:
        return tuple(str(entry.get(field_name, "")) for entry in self.characteristics())


def fetch_bytes(
    url: str,
    *,
    limit: int = MAX_SERIES_MATRIX_BYTES,
    attempts: int = NETWORK_ATTEMPTS,
    byte_budget: int | None = None,
    deadline_seconds: float | None = None,
) -> bytes:
    """Fetch a URL, resuming a cut transfer and falling back to a second client.

    The transport is resolved in one place so that a caller cannot accidentally take a different
    route from the one the release documents. Two things are handled here. First, the host these
    resources are read from advertises byte ranges and closes a long connection part-way through
    under load, so each request asks for the remaining bytes with a range header and appends what it
    receives: a transfer that is cut is resumed rather than restarted. ``byte_budget`` stops the
    transfer early, which is all the metadata block preceding a series matrix table needs. Second,
    some environments cannot complete a TLS handshake to this host through the standard library even
    though the host is reachable, so the same request is retried with the command line client that
    the environment does route; whichever transport answered is reported by the caller's evidence.
    """
    budget = byte_budget if byte_budget is not None else limit
    deadline = None if deadline_seconds is None else time.monotonic() + deadline_seconds
    try:
        return _fetch_streaming(
            url, budget=budget, limit=limit, attempts=attempts, deadline=deadline
        )
    except ResourceError as failure:
        get_logger("supply").warning("standard library transport failed for %s, retrying", url)
        if deadline is not None and time.monotonic() >= deadline:
            raise
        return _fetch_with_client(
            url, budget=budget, limit=limit, previous=failure, deadline=deadline
        )


def _fetch_streaming(
    url: str, *, budget: int, limit: int, attempts: int, deadline: float | None = None
) -> bytes:
    payload = bytearray()
    failures = 0
    while len(payload) <= limit:
        if len(payload) >= budget:
            break
        if deadline is not None and time.monotonic() >= deadline:
            raise ResourceError(f"{url} did not answer inside the time budget")
        headers = {"User-Agent": USER_AGENT, "Range": f"bytes={len(payload)}-"}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
                status = int(getattr(response, "status", 200))
                if status == 200:
                    payload = bytearray()
                chunk: bytes = response.read(READ_CHUNK_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
            failures = 0
        except Exception as failure:  # network boundary: resumed, then reported
            failures += 1
            if failures >= max(1, attempts):
                raise ResourceError(f"cannot reach {url}: {failure}") from failure
            time.sleep(1.5 * failures)
    if len(payload) > limit:
        raise ResourceError(f"{url} exceeds the {limit} byte ceiling")
    if not payload:
        raise ResourceError(f"{url} returned no bytes")
    return bytes(payload)


def _fetch_with_client(
    url: str,
    *,
    budget: int,
    limit: int,
    previous: Exception,
    deadline: float | None = None,
) -> bytes:
    """Fetch with the command line client the environment routes, as a second transport.

    The argument list is built directly rather than through a shell, so nothing in the URL is
    interpreted. The ceiling is enforced on what comes back, and if this route fails as well the
    error from the first route is reported, because that is the one a caller can act on.
    """
    client = shutil.which("curl")
    if client is None:
        raise previous
    # A host that is unreachable should fail quickly rather than hold the run open, so both the
    # connection and the whole transfer carry a short bound.
    budget_seconds: float = float(CLIENT_TIMEOUT_SECONDS)
    if deadline is not None:
        budget_seconds = max(1.0, min(budget_seconds, deadline - time.monotonic()))

    # A range that runs past the end of the resource returns the whole resource, so one command
    # covers both the bounded header read and the full download.
    chunk = min(budget, limit)
    command = [
        client,
        "--silent",
        "--show-error",
        "--location",
        "--connect-timeout",
        str(CONNECT_TIMEOUT_SECONDS),
        "--max-time",
        str(int(round(budget_seconds))),
        "--user-agent",
        USER_AGENT,
        "--range",
        f"0-{chunk - 1}",
        "--output",
        "-",
        url,
    ]
    completed = subprocess.run(command, capture_output=True, check=False, timeout=600)
    payload = completed.stdout
    if completed.returncode != 0 or not payload:
        raise previous
    if len(payload) > limit:
        raise ResourceError(f"{url} exceeds the {limit} byte ceiling")
    return payload


def _decode(payload: bytes, url: str, *, partial: bool = False) -> str:
    if url.endswith(".gz"):
        try:
            return gzip.decompress(payload).decode("utf-8", errors="replace")
        except OSError as failure:
            if not partial:
                raise ResourceError(f"payload from {url} is not gzip") from failure
            return _decode_partial(payload)
    return payload.decode("utf-8", errors="replace")


def _decode_partial(payload: bytes) -> str:
    """Decode as much of a truncated gzip stream as the decompressor will give up."""
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    try:
        chunk: bytes = decompressor.decompress(payload)
    except zlib.error:
        return ""
    return chunk.decode("utf-8", errors="replace")


def parse_series_matrix(
    text: str, accession: str, source_url: str, sha256: str, *, partial: bool = False
) -> GeoSeries:
    """Parse the header metadata and, when it is present, the expression table."""
    sample_ids: tuple[str, ...] = ()
    titles: tuple[str, ...] = ()
    characteristic_fields: tuple[str, ...] = ()
    characteristic_rows: list[tuple[str, ...]] = []
    feature_ids: list[str] = []
    rows: list[list[float]] = []
    in_table = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("!Sample_geo_accession"):
            sample_ids = _split_quoted(line)
        elif line.startswith("!Sample_title"):
            titles = _split_quoted(line)
        elif line.startswith("!Sample_characteristics_ch1"):
            values = _split_quoted(line)
            if not characteristic_fields:
                characteristic_fields = _field_names(values)
            characteristic_rows.append(values)
        elif line.startswith("!series_matrix_table_begin"):
            in_table = True
        elif line.startswith("!series_matrix_table_end"):
            in_table = False
        elif in_table and line and not line.startswith("!"):
            cells = line.split("\t")
            if cells[0].strip('"') == "ID_REF":
                continue
            feature_ids.append(cells[0].strip('"'))
            rows.append([_as_float(cell) for cell in cells[1:]])

    characteristics = _characteristic_maps(
        characteristic_fields, characteristic_rows, len(sample_ids)
    )
    matrix = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(sample_ids)))
    return GeoSeries(
        accession=accession,
        sample_ids=sample_ids,
        titles=titles,
        characteristic_fields=characteristic_fields,
        characteristics=characteristics,
        feature_ids=tuple(feature_ids),
        matrix=matrix,
        source_url=source_url,
        sha256=sha256,
        partial=partial,
    )


def _field_names(values: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        if ":" in value:
            names.append(value.split(":", 1)[0].strip())
        else:
            names.append(f"annotation_{len(names)}")
    return tuple(names)


def _split_quoted(line: str) -> tuple[str, ...]:
    _, _, payload = line.partition("\t")
    cells = payload.split("\t")
    return tuple(cell.strip().strip('"') for cell in cells)


def _as_float(cell: str) -> float:
    stripped = cell.strip().strip('"')
    if stripped in {"", "NA", "null", "NaN"}:
        return float("nan")
    try:
        return float(stripped)
    except ValueError:
        return float("nan")


def _characteristic_maps(
    fields: tuple[str, ...],
    rows: list[tuple[str, ...]],
    n_samples: int,
) -> tuple[Mapping[str, str], ...]:
    if not rows:
        return tuple({} for _ in range(n_samples))
    tables: list[dict[str, str]] = [{} for _ in range(n_samples)]
    for row in rows:
        for position, cell in enumerate(row[:n_samples]):
            key, separator, value = cell.partition(":")
            fallback = fields[position] if position < len(fields) else "annotation"
            name = key.strip() if separator else fallback
            tables[position][name] = value.strip()
    return tuple(tables)


def load_series(
    accession: str,
    cache_dir: Path,
    *,
    header_only: bool = False,
    attempts: int = NETWORK_ATTEMPTS,
    deadline_seconds: float | None = None,
) -> GeoSeriesBundle:
    """Read every matrix of a series, downloading it once and digesting it.

    A series submitted on more than one platform has one matrix per platform and no combined matrix,
    so the bundle is built from all of them and the total sample count is what is checked against
    Table 5. ``header_only`` reads the metadata block alone: that is enough to count the samples,
    avoids transferring tens of megabytes for a table the check does not read, and the returned
    bundle records that it came from a partial transfer.
    """
    entry: DatasetResource = resource(accession)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed: list[GeoSeries] = []
    for url in entry.matrix_urls():
        file_name = url.rsplit("/", 1)[-1]
        target = cache_dir / file_name
        if target.is_file():
            payload = target.read_bytes()
        else:
            payload = fetch_bytes(
                url,
                byte_budget=HEADER_BUDGET_BYTES if header_only else None,
                attempts=attempts,
                deadline_seconds=deadline_seconds,
            )
            if not header_only:
                target.write_bytes(payload)
        digest = digest_bytes(payload)
        text = _decode(payload, url, partial=header_only)
        parsed.append(parse_series_matrix(text, accession, url, digest, partial=header_only))
    bundle = GeoSeriesBundle(accession=accession, platforms=tuple(parsed))
    if bundle.n_samples() != entry.n_samples:
        raise ResourceError(
            f"{accession} carries {bundle.n_samples()} samples across {len(parsed)} platform "
            f"matrices, Table 5 reports {entry.n_samples}"
        )
    return bundle


def discover_matrix_files(accession: str) -> tuple[str, ...]:
    """List the matrix directory of a series and return the file names it holds."""
    entry = resource(accession)
    payload = fetch_bytes(entry.matrix_dir_url, limit=1 << 20)
    text = payload.decode("utf-8", errors="replace")
    names: list[str] = []
    for token in text.split('href="'):
        candidate = token.split('"', 1)[0]
        if candidate.endswith("_series_matrix.txt.gz"):
            names.append(candidate)
    return tuple(sorted(names))


def endpoint_text(series: GeoSeriesBundle) -> str:
    """Concatenate the annotation fields that could carry the response endpoint."""
    joined = " ".join(
        f"{key} {value}" for entry in series.characteristics() for key, value in entry.items()
    )
    return f"{' '.join(series.titles())} {joined}".lower()


def endpoint_signature(series: GeoSeriesBundle) -> Mapping[str, int]:
    """How often each endpoint keyword appears in the annotation text."""
    text = endpoint_text(series)
    counts: dict[str, int] = {}
    for endpoint, keywords in ENDPOINT_KEYWORDS.items():
        counts[endpoint] = sum(text.count(needle) for needle in keywords)
    return counts


def matches_reported_endpoint(series: GeoSeriesBundle, endpoint: str) -> bool:
    """Whether the series annotation mentions the endpoint Table 5 assigns to it."""
    needles = ENDPOINT_KEYWORDS.get(endpoint)
    if needles is None:
        return False
    text = endpoint_text(series)
    return any(needle in text for needle in needles)


def query_gdc_project(
    project_id: str,
    fields: str = "project_id,summary.case_count",
    *,
    attempts: int = NETWORK_ATTEMPTS,
    deadline_seconds: float | None = None,
) -> Mapping[str, object]:
    """Read the case count of an open-tier GDC project."""
    entry = resource(project_id)
    url = f"{entry.download_url}?fields={fields}"
    payload = fetch_bytes(
        url,
        limit=4 * 1024 * 1024,
        attempts=attempts,
        deadline_seconds=deadline_seconds,
    )
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except json.JSONDecodeError as failure:
        raise ResourceError(f"{url} did not return JSON") from failure
    if not isinstance(parsed, Mapping):
        raise ResourceError(f"{url} did not return a JSON object")
    data = parsed.get("data")
    if not isinstance(data, Mapping):
        raise ResourceError(f"{url} carried no data block")
    summary = data.get("summary")
    case_count = summary.get("case_count") if isinstance(summary, Mapping) else None
    return {
        "project_id": data.get("project_id", project_id),
        "case_count": int(case_count) if isinstance(case_count, int) else 0,
    }


def series_to_matrix(bundle: GeoSeriesBundle) -> Mapping[str, np.ndarray]:
    """Samples by features for each platform matrix of a series.

    A series submitted on two platforms yields two entries; they are kept apart rather than
    concatenated, because probe identifiers of one platform do not index the other, and the sample
    axes overlap only in part.
    """
    blocks: dict[str, np.ndarray] = {}
    for series in bundle.platforms:
        if series.matrix.size == 0:
            continue
        transposed: np.ndarray = np.asarray(series.matrix, dtype=np.float64).T
        blocks[series.source_url.rsplit("/", 1)[-1]] = np.nan_to_num(transposed, nan=0.0)
    return blocks


@dataclass
class ResourceLedger:
    """Bookkeeping for a batch of reads, written next to the tables it feeds."""

    entries: dict[str, dict[str, object]] = field(default_factory=dict)

    def note(self, accession: str, **values: object) -> None:
        bucket = self.entries.setdefault(accession, {})
        bucket.update(values)

    def as_dict(self) -> Mapping[str, Mapping[str, object]]:
        return {key: dict(value) for key, value in self.entries.items()}


def read_series_matrix_text(payload: bytes) -> str:
    """Decode an already-fetched series matrix, for callers that fetch elsewhere."""
    buffer = io.BytesIO(payload)
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(buffer.read()).decode("utf-8", errors="replace")
    return buffer.getvalue().decode("utf-8", errors="replace")
