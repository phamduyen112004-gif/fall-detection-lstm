"""Advanced fall alert system with fault tolerance and Telegram integration."""

from __future__ import annotations

import os
import queue
import threading
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (IDX_LEFT_ANKLE, IDX_LEFT_HIP, IDX_RIGHT_ANKLE,
                        IDX_RIGHT_HIP, N_CHANNELS, N_KEYPOINTS, SEQ_LEN)


# COCO 17-keypoint skeleton edges
SKELETON_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


def _is_missing_frame(pose_frame: np.ndarray) -> bool:
    if pose_frame.shape != (N_KEYPOINTS, N_CHANNELS):
        return True
    xy_missing = np.isnan(pose_frame[:, :2]).all()
    conf_missing = np.nan_to_num(pose_frame[:, 2], nan=0.0).max() <= 0.0
    return bool(xy_missing or conf_missing)


def interpolate_short_missing_runs(
    pose_seq: np.ndarray,
    max_missing_frames: int = 3,
) -> np.ndarray:
    """
    Repair short missing runs (<=3 frames) with linear interpolation.
    Missing frame is identified when all keypoints are absent/zero-confidence.
    """
    repaired = pose_seq.copy().astype(np.float32)
    T = repaired.shape[0]
    missing = np.array([_is_missing_frame(repaired[t]) for t in range(T)], dtype=bool)

    start = 0
    while start < T:
        if not missing[start]:
            start += 1
            continue
        end = start
        while end < T and missing[end]:
            end += 1

        run_len = end - start
        left = start - 1
        right = end
        if (
            run_len <= max_missing_frames
            and left >= 0
            and right < T
            and not missing[left]
            and not missing[right]
        ):
            steps = run_len + 1
            for i in range(run_len):
                alpha = (i + 1) / steps
                repaired[start + i] = (1.0 - alpha) * repaired[left] + alpha * repaired[right]
        start = end
    return repaired


def render_skeleton_privacy_frame(
    pose_frame: np.ndarray,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Draw skeleton over black background for privacy-preserving alert image."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if pose_frame.shape != (N_KEYPOINTS, N_CHANNELS):
        return canvas

    x = np.clip((pose_frame[:, 0] * width).astype(np.int32), 0, width - 1)
    y = np.clip((pose_frame[:, 1] * height).astype(np.int32), 0, height - 1)
    c = np.nan_to_num(pose_frame[:, 2], nan=0.0)

    for i, j in SKELETON_EDGES:
        if c[i] > 0.05 and c[j] > 0.05:
            cv2.line(canvas, (x[i], y[i]), (x[j], y[j]), (0, 255, 255), 2)
    for k in range(N_KEYPOINTS):
        if c[k] > 0.05:
            cv2.circle(canvas, (x[k], y[k]), 3, (0, 255, 0), -1)
    return canvas


def _mean_hips_and_feet_y(pose_frame: np.ndarray) -> Tuple[float, float]:
    hips_y = float((pose_frame[IDX_LEFT_HIP, 1] + pose_frame[IDX_RIGHT_HIP, 1]) / 2.0)
    feet_y = float((pose_frame[IDX_LEFT_ANKLE, 1] + pose_frame[IDX_RIGHT_ANKLE, 1]) / 2.0)
    return hips_y, feet_y


@dataclass
class FallDecisionState:
    waiting_confirmation: bool = False
    remaining_frames: int = 0
    hips_y_history: Optional[list] = None
    feet_y_history: Optional[list] = None
    impact_prob: float = 0.0
    impact_pose: Optional[np.ndarray] = None
    peak_acc: float = 0.0


class TelegramAlertClient:
    """Telegram alert sender using HTTPS requests."""

    def __init__(self, bot_token: str, chat_id: str, timeout_sec: int = 10):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_sec = timeout_sec
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_alert(self, text: str, image_bgr: np.ndarray) -> None:
        send_msg_url = f"{self.base_url}/sendMessage"
        requests.post(
            send_msg_url,
            data={"chat_id": self.chat_id, "text": text},
            timeout=self.timeout_sec,
        )

        send_photo_url = f"{self.base_url}/sendPhoto"
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            return
        requests.post(
            send_photo_url,
            data={"chat_id": self.chat_id},
            files={"photo": ("skeleton_alert.jpg", encoded.tobytes(), "image/jpeg")},
            timeout=self.timeout_sec,
        )


def _load_telegram_credentials() -> Tuple[str, str]:
    """
    Load Telegram credentials with fallback priority:
    1) .env / environment variables
    2) Kaggle user_secrets (if running on Kaggle)
    """
    load_dotenv(override=False)
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if bot_token and chat_id:
        return bot_token, chat_id

    # Kaggle fallback
    is_kaggle = bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE") or os.getenv("KAGGLE_URL_BASE"))
    if is_kaggle:
        try:
            from kaggle_secrets import UserSecretsClient

            client = UserSecretsClient()
            bot_token = client.get_secret("TELEGRAM_BOT_TOKEN") or ""
            chat_id = client.get_secret("TELEGRAM_CHAT_ID") or ""
            bot_token = bot_token.strip()
            chat_id = chat_id.strip()
            if bot_token and chat_id:
                return bot_token, chat_id
        except Exception:
            # Keep service alive even if kaggle_secrets is unavailable.
            pass

    return "", ""


class AdvancedAlertSystem:
    """
    Threaded post-processing for real-time fall confirmation.

    Workflow:
    - receive probability + pose frame stream,
    - keep 75-frame deque for model continuity,
    - interpolate short missing runs (<=3),
    - double-threshold confirmation:
      1) impact probability > 0.8 -> waiting state
      2) 2s stationary check (variance of hips y + hips_y > feet_y)
    - send privacy-preserving skeleton alert to Telegram.
    """

    def __init__(
        self,
        location: str = "Unknown room",
        upper_threshold: float = 0.8,
        stationary_frames: int = 50,
        stationary_var_threshold: float = 2e-4,
        hip_rise_cancel_threshold: float = 0.06,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ):
        self.location = location
        self.upper_threshold = upper_threshold
        self.stationary_frames = stationary_frames
        self.stationary_var_threshold = stationary_var_threshold
        self.hip_rise_cancel_threshold = hip_rise_cancel_threshold

        loaded_bot_token, loaded_chat_id = _load_telegram_credentials()
        self.telegram_bot_token = (telegram_bot_token or loaded_bot_token).strip()
        self.telegram_chat_id = (telegram_chat_id or loaded_chat_id).strip()
        self.telegram_client = None
        if self.telegram_bot_token and self.telegram_chat_id:
            self.telegram_client = TelegramAlertClient(self.telegram_bot_token, self.telegram_chat_id)
        else:
            print("Thiếu cấu hình Telegram, chức năng cảnh báo bị tắt")

        self.pose_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        self.packet_queue: "queue.Queue[Dict]" = queue.Queue(maxsize=1024)
        self.stop_event = threading.Event()
        self.state = FallDecisionState(
            waiting_confirmation=False,
            remaining_frames=0,
            hips_y_history=[],
            feet_y_history=[],
        )
        self.worker = threading.Thread(target=self._worker_loop, daemon=True, name="alert-postprocess-worker")

    def start(self) -> None:
        if not self.worker.is_alive():
            self.worker.start()

    def stop(self, timeout_sec: float = 2.0) -> None:
        self.stop_event.set()
        try:
            self.packet_queue.put_nowait({"_stop": True})
        except queue.Full:
            pass
        self.worker.join(timeout=timeout_sec)

    def submit(
        self,
        probability: float,
        pose_frame: np.ndarray,
        peak_acceleration: float = 0.0,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Non-blocking push for realtime stream.
        pose_frame must be [17,3] normalized keypoints.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        packet = {
            "prob": float(probability),
            "pose": pose_frame.astype(np.float32),
            "peak_acc": float(peak_acceleration),
            "timestamp": timestamp,
        }
        try:
            self.packet_queue.put_nowait(packet)
        except queue.Full:
            # Drop oldest-like behavior: skip packet to preserve realtime.
            pass

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            packet = self.packet_queue.get()
            if packet.get("_stop"):
                break
            self._process_packet(packet)

    def _process_packet(self, packet: Dict) -> None:
        pose = packet["pose"]
        prob = float(packet["prob"])
        peak_acc = float(packet["peak_acc"])
        timestamp = packet["timestamp"]

        self.pose_buffer.append(pose)
        if len(self.pose_buffer) < SEQ_LEN:
            return

        seq = np.stack(self.pose_buffer, axis=0)  # [75,17,3]
        seq = interpolate_short_missing_runs(seq, max_missing_frames=3)
        latest_pose = seq[-1]
        hips_y, feet_y = _mean_hips_and_feet_y(latest_pose)

        st = self.state
        if not st.waiting_confirmation:
            if prob > self.upper_threshold:
                st.waiting_confirmation = True
                st.remaining_frames = self.stationary_frames
                st.hips_y_history = [hips_y]
                st.feet_y_history = [feet_y]
                st.impact_prob = prob
                st.impact_pose = latest_pose.copy()
                st.peak_acc = peak_acc
            return

        # waiting for stationary confirmation
        st.hips_y_history.append(hips_y)
        st.feet_y_history.append(feet_y)
        st.remaining_frames -= 1

        # Cancel if hips rise suddenly (as requested)
        if len(st.hips_y_history) >= 2:
            hip_delta = st.hips_y_history[-1] - st.hips_y_history[0]
            if hip_delta > self.hip_rise_cancel_threshold:
                self._reset_state()
                return

        if st.remaining_frames > 0:
            return

        hips_var = float(np.var(np.array(st.hips_y_history, dtype=np.float32)))
        hips_mean = float(np.mean(np.array(st.hips_y_history, dtype=np.float32)))
        feet_mean = float(np.mean(np.array(st.feet_y_history, dtype=np.float32)))

        # Confirmation rule from requirement:
        # variance ~ 0 and y_hips > y_feet
        is_stationary = hips_var <= self.stationary_var_threshold
        lying_condition = hips_mean > feet_mean
        if is_stationary and lying_condition:
            self._trigger_alert(timestamp=timestamp)
        self._reset_state()

    def _trigger_alert(self, timestamp: str) -> None:
        st = self.state
        impact_pose = st.impact_pose if st.impact_pose is not None else np.zeros((N_KEYPOINTS, N_CHANNELS), dtype=np.float32)
        privacy_img = render_skeleton_privacy_frame(impact_pose)
        text = (
            f"🚨 CANH BAO NGUY HIEM: Phat hien nguoi nga tai [{self.location}] vao luc [{timestamp}]. "
            f"Muc do va cham (gia toc cao nhat): [{st.peak_acc:.4f}]."
        )
        if self.telegram_client is not None:
            try:
                self.telegram_client.send_alert(text=text, image_bgr=privacy_img)
            except requests.RequestException:
                # Keep system resilient in case Telegram is unreachable.
                pass

    def _reset_state(self) -> None:
        self.state.waiting_confirmation = False
        self.state.remaining_frames = 0
        self.state.hips_y_history = []
        self.state.feet_y_history = []
        self.state.impact_prob = 0.0
        self.state.impact_pose = None
        self.state.peak_acc = 0.0


def example_usage():
    """
    Minimal integration sketch:

    alert_sys = AdvancedAlertSystem(location="Home_01")
    alert_sys.start()
    ...
    # inside realtime loop
    alert_sys.submit(probability=fall_prob, pose_frame=pose_17x3, peak_acceleration=max_acc)
    ...
    alert_sys.stop()
    """
    pass

