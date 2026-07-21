from __future__ import annotations

from unittest.mock import patch

import torch

from learning.karate_selfcal.stage_a.dataset import KarateStageAClipDataset


class _FakeFrameDataset:
    def __init__(self, entries: list[dict[str, object]]):
        self.entries = entries

    def __getitem__(self, index: int) -> dict[str, object]:
        entry = self.entries[index]
        return {
            "ray_tokens": torch.full((17, 4, 11), float(index)),
            "view_mask": torch.ones(4, dtype=torch.bool),
            "joint_view_mask": torch.ones((17, 4), dtype=torch.bool),
            "camera_origin_delta": torch.zeros((4, 3)),
            "camera_translation_delta": torch.zeros((4, 3)),
            "camera_rotation_delta": torch.zeros((4, 3)),
            "camera_scale_delta": torch.zeros(4),
            "camera_origin_delta_valid": torch.ones(4, dtype=torch.bool),
            "sample_id": str(entry["id"]),
            "sequence": str(entry["sequence"]),
            "frame_id": int(entry["frame_id"]),
            "person_id": str(entry["person_id"]),
        }


def test_clip_dataset_keeps_people_in_separate_temporal_tracks() -> None:
    entries = [
        {
            "id": f"person_{person_id}_frame_{frame_id}",
            "sequence": "karate_001",
            "variant": 0,
            "person_id": person_id,
            "frame_id": frame_id,
        }
        for person_id in ("fighter_a", "fighter_b")
        for frame_id in range(4)
    ]
    fake_frames = _FakeFrameDataset(entries)

    with patch(
        "learning.karate_selfcal.stage_a.dataset.KarateStageADataset",
        return_value=fake_frames,
    ):
        dataset = KarateStageAClipDataset(
            "unused.json",
            split="train",
            clip_length=3,
            clip_stride=2,
        )

    assert len(dataset) == 4
    for clip_index in range(len(dataset)):
        clip = dataset[clip_index]
        person_id = clip["person_id"]
        frame_indexes = dataset.clips[clip_index][1]
        assert {entries[index]["person_id"] for index in frame_indexes} == {person_id}
        assert person_id in clip["sample_id"]


if __name__ == "__main__":
    test_clip_dataset_keeps_people_in_separate_temporal_tracks()
