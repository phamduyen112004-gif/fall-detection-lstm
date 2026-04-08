"""Window-level and event-level evaluation utilities."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


@dataclass
class Event:
    start: int
    end: int


def _extract_events(binary_seq: np.ndarray) -> List[Event]:
    events: List[Event] = []
    in_event = False
    start = 0
    for i, v in enumerate(binary_seq.astype(int)):
        if v == 1 and not in_event:
            in_event = True
            start = i
        if v == 0 and in_event:
            in_event = False
            events.append(Event(start=start, end=i - 1))
    if in_event:
        events.append(Event(start=start, end=len(binary_seq) - 1))
    return events


def evaluate_window_and_event_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fps: float = 25.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = y_true.reshape(-1).astype(int)
    y_pred = (y_prob.reshape(-1) >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    specificity = tn / max(tn + fp, 1)

    true_events = _extract_events(y_true)
    pred_events = _extract_events(y_pred)
    matched_pred = set()
    delays = []
    event_tp = 0
    for te in true_events:
        found = False
        for j, pe in enumerate(pred_events):
            if j in matched_pred:
                continue
            overlap = not (pe.end < te.start or pe.start > te.end)
            if overlap:
                matched_pred.add(j)
                found = True
                event_tp += 1
                delays.append(max(0, pe.start - te.start) / fps)
                break
        if not found:
            delays.append(float("nan"))

    event_fn = len(true_events) - event_tp
    event_fp = len(pred_events) - len(matched_pred)
    total_hours = len(y_true) / fps / 3600.0
    false_alarm_per_hour = event_fp / max(total_hours, 1e-9)
    valid_delays = [d for d in delays if np.isfinite(d)]
    mean_delay = float(np.mean(valid_delays)) if valid_delays else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "event_tp": float(event_tp),
        "event_fp": float(event_fp),
        "event_fn": float(event_fn),
        "false_alarm_per_hour": float(false_alarm_per_hour),
        "detection_delay_sec": float(mean_delay) if np.isfinite(mean_delay) else -1.0,
    }
