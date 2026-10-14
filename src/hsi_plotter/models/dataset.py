from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Modality(StrEnum):
    VIS = "vis"
    NIR = "nir"
    REDNIR = "rednir"
    UNSPECIFIED = "unspecified"


_MODALITY_PREFIXES = (
    ("rednir-", Modality.REDNIR),
    ("nir-", Modality.NIR),
    ("vis-", Modality.VIS),
)

MODALITY_DIRECTORY_NAMES = {
    "vis": Modality.VIS,
    "visible": Modality.VIS,
    "nir": Modality.NIR,
    "infrared": Modality.NIR,
    "rednir": Modality.REDNIR,
    "red-nir": Modality.REDNIR,
}

MODALITY_LABELS = {
    Modality.VIS: "VIS",
    Modality.NIR: "NIR",
    Modality.REDNIR: "RedNIR",
    Modality.UNSPECIFIED: "未指定",
}


def parse_sequence_identity(filename_stem: str) -> tuple[Modality, str]:
    normalized = filename_stem.strip()
    lowered = normalized.casefold()
    for prefix, modality in _MODALITY_PREFIXES:
        if lowered.startswith(prefix):
            return modality, normalized[len(prefix) :]
    return Modality.UNSPECIFIED, normalized


@dataclass(frozen=True, slots=True)
class SequenceKey:
    dataset: str
    modality: Modality
    name: str

    @property
    def identifier(self) -> str:
        return f"{self.dataset}:{self.modality.value}:{self.name}"
