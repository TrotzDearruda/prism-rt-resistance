"""Metabolite vocabulary and its metabolically coherent blocks.

Ref: Methods Sec. 6.1 (M lies in R^q with q the vocabulary size) and Sec. 6.6 (sparsity
is imposed by a group penalty over metabolically coherent blocks, since an element-wise
penalty selects isolated metabolites rather than a programme). The manuscript does not
print the vocabulary, so the entries below are the named intermediates of the six
programmes it does describe.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from prism.errors import CohortError

GLYCOLYTIC: tuple[str, ...] = (
    "glucose",
    "glucose-6-phosphate",
    "fructose-6-phosphate",
    "fructose-1,6-bisphosphate",
    "dihydroxyacetone-phosphate",
    "glyceraldehyde-3-phosphate",
    "1,3-bisphosphoglycerate",
    "3-phosphoglycerate",
    "2-phosphoglycerate",
    "phosphoenolpyruvate",
    "pyruvate",
    "lactate",
)

GLUTAMINOLYTIC: tuple[str, ...] = (
    "glutamine",
    "glutamate",
    "alpha-ketoglutarate",
    "succinyl-coenzyme-a",
    "succinate",
    "fumarate",
    "malate",
    "aspartate",
)

LIPID_DESATURATION: tuple[str, ...] = (
    "acetyl-coenzyme-a",
    "malonyl-coenzyme-a",
    "palmitoyl-coenzyme-a",
    "palmitate",
    "palmitoleate",
    "stearoyl-coenzyme-a",
    "oleate",
    "arachidonate",
)

HYPOXIA_REDOX: tuple[str, ...] = (
    "2-hydroxyglutarate",
    "itaconate",
    "hypotaurine",
    "glycerophosphocholine",
    "phosphocholine",
    "choline",
)

NUCLEOTIDE: tuple[str, ...] = (
    "ribose-5-phosphate",
    "phosphoribosyl-pyrophosphate",
    "inosine-monophosphate",
    "adenosine-monophosphate",
    "guanosine-monophosphate",
    "cytidine-monophosphate",
    "uridine-monophosphate",
    "thymidine-monophosphate",
)

REDOX_BUFFERING: tuple[str, ...] = (
    "cysteine",
    "cystine",
    "gamma-glutamylcysteine",
    "glutathione-reduced",
    "glutathione-disulfide",
    "methionine",
)

BLOCK_BY_PROGRAMME: Mapping[str, tuple[str, ...]] = {
    "P1": GLYCOLYTIC,
    "P2": GLUTAMINOLYTIC,
    "P3": LIPID_DESATURATION,
    "P4": HYPOXIA_REDOX,
    "P5": NUCLEOTIDE,
    "P6": REDOX_BUFFERING,
}

# Direction each entry is expected to move with respect to radiosensitivity. An upward
# shift of a saturated fatty acid tracks a less desaturated membrane, hence the negative
# entries inside the lipid block; the glutathione ratio moves the opposite way to its two
# components, so the disulfide carries the negative entry there.
NEGATIVE_ENTRIES: frozenset[str] = frozenset(
    {
        "glucose",
        "glucose-6-phosphate",
        "fructose-6-phosphate",
        "dihydroxyacetone-phosphate",
        "aspartate",
        "palmitate",
        "stearoyl-coenzyme-a",
        "glutathione-disulfide",
        "cystine",
    }
)


@dataclass(frozen=True)
class Metabolite:
    name: str
    index: int
    programme: str
    expected_sign: int

    def reflects_resistance_upward(self) -> bool:
        return self.expected_sign > 0


class Lexicon:
    """Ordered metabolite vocabulary with a partition into programme blocks."""

    def __init__(self, programme_ids: tuple[str, ...], blocks: tuple[tuple[int, ...], ...]) -> None:
        self._programme_ids = programme_ids
        self._blocks = blocks
        flat = [index for block in blocks for index in block]
        if sorted(flat) != list(range(len(flat))):
            raise CohortError("blocks must partition a contiguous index range")
        self._size = len(flat)
        self._index_of_block: dict[int, int] = {}
        for block_position, block in enumerate(blocks):
            for index in block:
                self._index_of_block[index] = block_position
        self._offset_of_block: dict[int, int] = {}
        cursor = 0
        for block_position, block in enumerate(blocks):
            self._offset_of_block[block_position] = cursor
            cursor += len(block)

    @classmethod
    def from_programmes(cls, programme_ids: tuple[str, ...]) -> Lexicon:
        """The named vocabulary, laid out in programme order."""
        blocks: list[tuple[int, ...]] = []
        cursor = 0
        for programme in programme_ids:
            try:
                members = BLOCK_BY_PROGRAMME[programme]
            except KeyError as failure:
                raise CohortError(f"no metabolite block is defined for {programme}") from failure
            blocks.append(tuple(range(cursor, cursor + len(members))))
            cursor += len(members)
        return cls(programme_ids, tuple(blocks))

    @property
    def size(self) -> int:
        return self._size

    def n_blocks(self) -> int:
        """Number of metabolically coherent blocks, one per programme."""
        return len(self._blocks)

    @property
    def blocks(self) -> tuple[tuple[int, ...], ...]:
        return self._blocks

    @property
    def programme_ids(self) -> tuple[str, ...]:
        return self._programme_ids

    def block_of(self, index: int) -> int:
        try:
            return self._index_of_block[index]
        except KeyError as failure:
            raise CohortError(f"metabolite index {index} is outside the vocabulary") from failure

    def programme_of(self, index: int) -> str:
        return self._programme_ids[self.block_of(index)]

    def block_members(self, programme: str) -> tuple[int, ...]:
        return self._blocks[self._programme_ids.index(programme)]

    def offset(self, block_position: int) -> int:
        return self._offset_of_block[block_position]

    def names(self) -> tuple[str, ...]:
        collected: list[str] = []
        for programme in self._programme_ids:
            collected.extend(BLOCK_BY_PROGRAMME[programme])
        return tuple(collected)

    def name_of(self, index: int) -> str:
        return self.names()[index]

    def catalogue(self) -> tuple[Metabolite, ...]:
        entries: list[Metabolite] = []
        for index, name in enumerate(self.names()):
            entries.append(
                Metabolite(
                    name=name,
                    index=index,
                    programme=self.programme_of(index),
                    expected_sign=-1 if name in NEGATIVE_ENTRIES else 1,
                )
            )
        return tuple(entries)

    def expected_signs(self) -> tuple[int, ...]:
        return tuple(entry.expected_sign for entry in self.catalogue())

    def block_sizes(self) -> tuple[int, ...]:
        return tuple(len(block) for block in self._blocks)

    def block_weights(self) -> tuple[float, ...]:
        """Group-penalty weights, set so that a long block is not penalised harder than
        the statement it carries."""
        sizes = self.block_sizes()
        largest = max(sizes)
        return tuple(float(largest) / float(size) for size in sizes)

    def index_blocks(self) -> tuple[tuple[int, ...], ...]:
        return self._blocks


def default_lexicon() -> Lexicon:
    return Lexicon.from_programmes(("P1", "P2", "P3", "P4", "P5", "P6"))


def supports_lexicon(vocabulary_size: int, programme_ids: tuple[str, ...]) -> bool:
    """Whether a candidate vocabulary can carry every programme block."""
    try:
        candidate = Lexicon.from_programmes(programme_ids)
    except CohortError:
        return False
    return candidate.size == vocabulary_size
