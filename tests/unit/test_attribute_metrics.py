from pathlib import Path

import pytest

from hsi_plotter.metrics.attributes import resolve_attribute_sequences
from hsi_plotter.models.attribute_evaluation import AttributeDefinition
from hsi_plotter.models.dataset import Modality, SequenceKey


def _key(modality: Modality, name: str) -> SequenceKey:
    return SequenceKey(dataset="project", modality=modality, name=name)


def test_resolve_attribute_sequences_supports_prefixed_and_unique_plain_names() -> None:
    vis_ball = _key(Modality.VIS, "Ball")
    nir_car = _key(Modality.NIR, "Car59")
    ground_truth = {vis_ball: Path("vis-ball.txt"), nir_car: Path("nir-car59.txt")}
    attributes = (
        AttributeDefinition("OCC", ("VIS-ball", "nir-CAR59")),
        AttributeDefinition("SV", ("ball",)),
    )

    resolution = resolve_attribute_sequences(attributes, ground_truth)

    assert resolution.keys_by_attribute["OCC"] == (vis_ball, nir_car)
    assert resolution.keys_by_attribute["SV"] == (vis_ball,)
    assert resolution.unknown_by_attribute == {}


def test_resolve_attribute_sequences_reports_unknown_and_ambiguous_plain_names() -> None:
    vis_ball = _key(Modality.VIS, "ball")
    nir_ball = _key(Modality.NIR, "ball")
    ground_truth = {vis_ball: Path("vis-ball.txt"), nir_ball: Path("nir-ball.txt")}
    attributes = (AttributeDefinition("OCC", ("ball", "rednir-missing")),)

    resolution = resolve_attribute_sequences(attributes, ground_truth)

    assert resolution.keys_by_attribute["OCC"] == ()
    assert resolution.unknown_by_attribute["OCC"] == ("ball", "rednir-missing")


def test_attribute_definition_rejects_blank_name() -> None:
    with pytest.raises(ValueError, match="属性名"):
        AttributeDefinition("  ", ("ball",))

