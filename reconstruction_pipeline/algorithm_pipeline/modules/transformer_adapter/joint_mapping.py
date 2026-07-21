from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from learning.rumpl_fourview import MODEL_JOINT_NAMES


JOINT_ID_TO_NAME = {index: name for index, name in enumerate(MODEL_JOINT_NAMES)}
MODEL_JOINT_NAMES_BY_INDEX = list(MODEL_JOINT_NAMES)


def person_keypoints_to_model_array(person: Dict[str, object]) -> np.ndarray:
    joints = np.zeros((len(MODEL_JOINT_NAMES), 3), dtype=np.float32)
    for kp in person.get("keypoints", []):
        joint_id = int(kp.get("id", -1))
        if 0 <= joint_id < len(MODEL_JOINT_NAMES):
            joints[joint_id, 0] = float(kp.get("x", 0.0))
            joints[joint_id, 1] = float(kp.get("y", 0.0))
            joints[joint_id, 2] = float(kp.get("confidence", 0.0))
    return joints


def select_primary_person(frame: Dict[str, object]) -> Optional[Dict[str, object]]:
    people = frame.get("people", [])
    if not people:
        return None
    return people[0]
