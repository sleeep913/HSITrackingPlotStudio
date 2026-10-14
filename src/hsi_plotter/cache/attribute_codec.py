"""JSON-safe codec for attribute evaluation results."""

from __future__ import annotations

from typing import Any

from hsi_plotter.models.attribute_evaluation import (
    AttributeDefinition,
    AttributeEvaluationRun,
    AttributeScore,
    TrackerAttributeResult,
)


def encode_attribute_run(run: AttributeEvaluationRun) -> dict[str, Any]:
    return {
        "codec_version": 1,
        "attributes": [
            {"name": item.name, "sequences": list(item.sequences)}
            for item in run.attributes
        ],
        "trackers": [
            {
                "tracker_id": tracker.tracker_id,
                "tracker_name": tracker.tracker_name,
                "scores": {
                    name: {
                        "auc": score.auc,
                        "dp": score.dp,
                        "resolved_count": score.resolved_count,
                        "succeeded_count": score.succeeded_count,
                        "failed_count": score.failed_count,
                    }
                    for name, score in tracker.scores.items()
                },
            }
            for tracker in run.trackers
        ],
        "unknown_by_attribute": {
            key: list(value) for key, value in run.unknown_by_attribute.items()
        },
        "issues": list(run.issues),
    }


def decode_attribute_run(payload: dict[str, Any]) -> AttributeEvaluationRun:
    if payload.get("codec_version") != 1:
        raise ValueError("不支持的属性缓存版本")
    attributes = tuple(
        AttributeDefinition(str(item["name"]), tuple(map(str, item["sequences"])))
        for item in payload["attributes"]
    )
    trackers = []
    for item in payload["trackers"]:
        scores = {
            str(name): AttributeScore(
                auc=value.get("auc"),
                dp=value.get("dp"),
                resolved_count=int(value["resolved_count"]),
                succeeded_count=int(value["succeeded_count"]),
                failed_count=int(value["failed_count"]),
            )
            for name, value in item["scores"].items()
        }
        trackers.append(
            TrackerAttributeResult(
                tracker_id=str(item["tracker_id"]),
                tracker_name=str(item["tracker_name"]),
                scores=scores,
            )
        )
    return AttributeEvaluationRun(
        attributes=attributes,
        trackers=tuple(trackers),
        unknown_by_attribute={
            str(key): tuple(map(str, value))
            for key, value in payload.get("unknown_by_attribute", {}).items()
        },
        issues=tuple(map(str, payload.get("issues", []))),
    )


__all__ = ["decode_attribute_run", "encode_attribute_run"]
