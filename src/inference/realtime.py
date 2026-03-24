"""Realtime post-processing for fall detection predictions."""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class SmartFallPostProcessor:
    """
    Smart realtime logic:
    1) Average probabilities over the latest 10 windows to smooth noise.
    2) Confirm fall with stationary check for next 1-2 seconds.
       If hip rises quickly again, cancel alert (person stood up).
    """

    fps: int = 25
    avg_window: int = 10
    prob_threshold: float = 0.5
    stationary_seconds: float = 2.0
    hip_rise_threshold: float = 0.08

    probs: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    in_confirmation: bool = False
    confirm_remaining_frames: int = 0
    hip_baseline: Optional[float] = None
    pending_prob: float = 0.0

    def _mean_hip_y(self, sequence: np.ndarray) -> float:
        """
        sequence shape: [75, K]
        Assumption:
        - Raw pose is flattened first in feature vector as [17 x 3] = 51 dims.
        - Left hip y at index (11*3 + 1), right hip y at index (12*3 + 1).
        """
        if sequence.ndim != 2 or sequence.shape[1] < 38:
            raise ValueError("Sequence must have shape [T, K] with K >= 38.")
        left_hip_y = sequence[:, 34]
        right_hip_y = sequence[:, 37]
        return float((left_hip_y.mean() + right_hip_y.mean()) / 2.0)

    def update(self, prob: float, sequence: np.ndarray) -> Tuple[int, float, str]:
        """
        Update with one new model probability and its feature sequence.
        Returns:
        - final_alert: 1 (fall) or 0 (normal)
        - avg_prob_10: averaged probability over last 10 predictions
        - status: 'normal' | 'candidate_fall' | 'confirmed_fall' | 'cancelled_fall'
        """
        self.probs.append(float(prob))
        avg_prob_10 = float(np.mean(self.probs))
        hip_now = self._mean_hip_y(sequence)

        if not self.in_confirmation:
            if len(self.probs) == self.avg_window and avg_prob_10 > self.prob_threshold:
                self.in_confirmation = True
                self.confirm_remaining_frames = max(1, int(self.stationary_seconds * self.fps))
                self.hip_baseline = hip_now
                self.pending_prob = avg_prob_10
                return 0, avg_prob_10, "candidate_fall"
            return 0, avg_prob_10, "normal"

        # Confirmation phase: if hip rises quickly => cancel alarm
        hip_rise = hip_now - (self.hip_baseline if self.hip_baseline is not None else hip_now)
        if hip_rise > self.hip_rise_threshold:
            self.in_confirmation = False
            self.confirm_remaining_frames = 0
            self.hip_baseline = None
            self.pending_prob = 0.0
            return 0, avg_prob_10, "cancelled_fall"

        self.confirm_remaining_frames -= 1
        if self.confirm_remaining_frames <= 0:
            self.in_confirmation = False
            self.hip_baseline = None
            if self.pending_prob > self.prob_threshold:
                return 1, avg_prob_10, "confirmed_fall"
            return 0, avg_prob_10, "normal"

        return 0, avg_prob_10, "candidate_fall"


def predict_realtime(
    model,
    sequence: np.ndarray,
    post_processor: SmartFallPostProcessor,
) -> Tuple[float, int, str]:
    """
    Predict one sequence and apply smart post-processing logic.
    Returns (raw_probability, final_alert, status).
    """
    raw_prob = float(model.predict(sequence[None, ...], verbose=0)[0, 0])
    final_alert, _, status = post_processor.update(raw_prob, sequence)
    return raw_prob, final_alert, status
