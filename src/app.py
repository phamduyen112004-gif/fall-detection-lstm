"""
Streamlit dashboard for testing fall detection on offline videos.

Dashboard nay giup kiem chung mo hinh tren video co san ma khong can webcam:
- Quan sat overlay skeleton + canh bao "FALL DETECTED!" theo thoi gian thuc.
- Theo doi cac chi so dong hoc (Vertical Velocity, Body Tilt Angle).
- Luu nhat ky su kien va false positives de phuc vu retraining.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import torch

try:
    from ultralytics import YOLO
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'ultralytics'. Install with `pip install ultralytics` "
        "or add it to your environment requirements."
    ) from e

from src.features.feature_engineering import compute_advanced_features, minmax_scale
from src.inference.alert_system import (SKELETON_EDGES, TelegramAlertClient,
                                        interpolate_short_missing_runs,
                                        render_skeleton_privacy_frame)
from src.inference.profiling import RuntimeProfiler
from src.models.architectures import TemporalAttention
from src.pose.smoothing import fill_and_smooth_window


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
FALSE_POS_DIR = ROOT_DIR / "data" / "false_positives"
MODELS_DIR = ROOT_DIR / "models"

SEQ_LEN = 75
FPS_DEFAULT = 25
UPPER_THRESHOLD = 0.8
STATIONARY_FRAMES = 50  # 2s at 25 FPS
STATIONARY_VAR_THRESHOLD = 2e-4


def _discover_test_videos() -> List[Path]:
    exts = {".mp4", ".avi", ".mov", ".mkv"}
    if not RAW_DATA_DIR.exists():
        return []
    return sorted([p for p in RAW_DATA_DIR.rglob("*") if p.suffix.lower() in exts])


def _extract_pose_from_result(result) -> Dict:
    pose = np.full((17, 3), np.nan, dtype=np.float32)
    pose[:, 2] = 0.0
    bbox = None
    if result.keypoints is None or len(result.keypoints) == 0:
        return {"pose": pose, "bbox": bbox}

    boxes = result.boxes
    if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
        person_idx = int(np.argmax(boxes.conf.cpu().numpy()))
        box_xyxy = boxes.xyxy[person_idx].cpu().numpy().astype(np.int32).tolist()
        bbox = tuple(box_xyxy)
    else:
        person_idx = 0

    xy = result.keypoints.xy[person_idx].cpu().numpy()
    conf_tensor = result.keypoints.conf
    conf = conf_tensor[person_idx].cpu().numpy() if conf_tensor is not None else np.ones((17,), dtype=np.float32)

    # Normalize using Le2i dimensions
    pose[:, 0] = xy[:, 0] / 320.0
    pose[:, 1] = xy[:, 1] / 240.0
    pose[:, 2] = conf
    return {"pose": pose, "bbox": bbox}


def _draw_overlay(frame: np.ndarray, pose: np.ndarray, bbox, prob: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    xs = np.clip((pose[:, 0] * w).astype(np.int32), 0, w - 1)
    ys = np.clip((pose[:, 1] * h).astype(np.int32), 0, h - 1)
    cs = np.nan_to_num(pose[:, 2], nan=0.0)

    for i, j in SKELETON_EDGES:
        if cs[i] > 0.05 and cs[j] > 0.05:
            cv2.line(out, (xs[i], ys[i]), (xs[j], ys[j]), (0, 255, 255), 2)
    for k in range(17):
        if cs[k] > 0.05:
            cv2.circle(out, (xs[k], ys[k]), 3, (0, 255, 0), -1)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        color = (0, 0, 255) if prob > UPPER_THRESHOLD else (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    cv2.putText(out, f"Confidence: {prob:.3f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 50), 2)
    if prob > UPPER_THRESHOLD:
        cv2.putText(out, "FALL DETECTED!", (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 0, 255), 3)
    return out


class InferenceWorker(threading.Thread):
    """Background thread for AI pipeline so Streamlit UI is less likely to freeze."""

    def __init__(self, yolo_model: YOLO, bilstm_model, send_telegram: bool, location: str, bot_token: str, chat_id: str):
        super().__init__(daemon=True)
        self.yolo_model = yolo_model
        self.bilstm_model = bilstm_model
        self.in_q: "queue.Queue[Optional[Dict]]" = queue.Queue(maxsize=16)
        self.out_q: "queue.Queue[Dict]" = queue.Queue(maxsize=64)
        self.pose_buffer: Deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        self.stop_event = threading.Event()
        self.send_telegram = send_telegram
        self.location = location
        self.telegram_client = TelegramAlertClient(bot_token, chat_id) if (send_telegram and bot_token and chat_id) else None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.profiler = RuntimeProfiler()

        # Confirmation state
        self.wait_confirm = False
        self.remain_confirm = 0
        self.hips_history: List[float] = []
        self.feet_history: List[float] = []
        self.impact_pose: Optional[np.ndarray] = None
        self.peak_acc: float = 0.0

    def run(self):
        while not self.stop_event.is_set():
            item = self.in_q.get()
            if item is None:
                break
            self._process(item)

    def stop(self):
        self.stop_event.set()
        try:
            self.in_q.put_nowait(None)
        except queue.Full:
            pass

    def submit(self, packet: Dict):
        try:
            self.in_q.put_nowait(packet)
        except queue.Full:
            pass

    def _process(self, packet: Dict):
        frame = packet["frame"]
        ts = packet["timestamp"]

        result = self.yolo_model.predict(frame, verbose=False, imgsz=320, conf=0.15, device=self.device, show=False)[0]
        parsed = _extract_pose_from_result(result)
        pose = parsed["pose"]
        bbox = parsed["bbox"]

        self.pose_buffer.append(pose)
        prob = 0.0
        body_angle = 0.0
        vy = 0.0
        peak_acc = 0.0
        event = None
        feature_seq = None

        if len(self.pose_buffer) == SEQ_LEN:
            seq = np.stack(self.pose_buffer, axis=0)
            seq = interpolate_short_missing_runs(seq, max_missing_frames=3)
            seq_smoothed = fill_and_smooth_window(seq)
            if seq_smoothed is None:
                seq_smoothed = seq

            feats = compute_advanced_features(seq_smoothed[None, ...]).astype(np.float32)  # [1,75,K]
            feats = minmax_scale(feats)
            feature_seq = feats[0]
            prob = float(self.bilstm_model.predict(feats, verbose=0)[0, 0])

            # Feature indices from compute_advanced_features:
            # body_angle -> 51, vy -> 57, ay -> 59
            body_angle = float(feats[0, -1, 51])
            vy = float(feats[0, -1, 57])
            peak_acc = float(np.max(np.abs(feats[0, :, 59])))
            event = self._check_confirmation(prob, seq_smoothed[-1], peak_acc, ts)

        frame_overlay = _draw_overlay(frame, pose, bbox, prob)
        self.profiler.update()
        perf = self.profiler.summary()
        out = {
            "timestamp": ts,
            "frame_overlay": frame_overlay,
            "pose": pose,
            "prob": prob,
            "vy": vy,
            "body_angle": body_angle,
            "peak_acc": peak_acc,
            "event": event,
            "feature_seq": feature_seq,
            "fps_avg": perf["fps_avg"],
            "ram_mb_avg": perf["ram_mb_avg"],
            "gpu_util_avg": perf["gpu_util_avg"],
        }
        try:
            self.out_q.put_nowait(out)
        except queue.Full:
            pass

    def _check_confirmation(self, prob: float, pose_frame: np.ndarray, peak_acc: float, ts: str):
        hips_y = float((pose_frame[11, 1] + pose_frame[12, 1]) / 2.0)
        feet_y = float((pose_frame[15, 1] + pose_frame[16, 1]) / 2.0)

        if not self.wait_confirm:
            if prob > UPPER_THRESHOLD:
                self.wait_confirm = True
                self.remain_confirm = STATIONARY_FRAMES
                self.hips_history = [hips_y]
                self.feet_history = [feet_y]
                self.impact_pose = pose_frame.copy()
                self.peak_acc = peak_acc
            return None

        self.hips_history.append(hips_y)
        self.feet_history.append(feet_y)
        self.remain_confirm -= 1

        # Cancel if hip rises again quickly
        if len(self.hips_history) >= 2 and (self.hips_history[-1] - self.hips_history[0]) > 0.06:
            self.wait_confirm = False
            self.remain_confirm = 0
            return None

        if self.remain_confirm > 0:
            return None

        hips_var = float(np.var(np.array(self.hips_history, dtype=np.float32)))
        hips_mean = float(np.mean(np.array(self.hips_history, dtype=np.float32)))
        feet_mean = float(np.mean(np.array(self.feet_history, dtype=np.float32)))
        confirmed = (hips_var <= STATIONARY_VAR_THRESHOLD) and (hips_mean > feet_mean)

        event = None
        if confirmed:
            skeleton_img = render_skeleton_privacy_frame(
                self.impact_pose if self.impact_pose is not None else pose_frame
            )
            event = {
                "timestamp": ts,
                "peak_acc": self.peak_acc,
                "hips_var": hips_var,
                "skeleton_img": skeleton_img,
            }
            if self.telegram_client is not None:
                try:
                    msg = (
                        f"🚨 CẢNH BÁO NGUY HIỂM: Phát hiện người ngã tại [{self.location}] vào lúc [{ts}]. "
                        f"Mức độ va chạm: [{self.peak_acc:.4f}]."
                    )
                    self.telegram_client.send_alert(msg, skeleton_img)
                except Exception:
                    pass

        self.wait_confirm = False
        self.remain_confirm = 0
        self.hips_history = []
        self.feet_history = []
        self.impact_pose = None
        self.peak_acc = 0.0
        return event


@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO("yolo11n-pose.pt")


@st.cache_resource(show_spinner=False)
def load_bilstm(model_path: str):
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"TemporalAttention": TemporalAttention},
        compile=False,
    )


def _save_false_positive(event: Dict, feature_seq: Optional[np.ndarray]):
    FALSE_POS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = FALSE_POS_DIR / f"fp_{stamp}.jpg"
    npy_path = FALSE_POS_DIR / f"fp_{stamp}.npy"
    cv2.imwrite(str(img_path), event["skeleton_img"])
    if feature_seq is not None:
        np.save(npy_path, feature_seq.astype(np.float32))


def main():
    st.set_page_config(page_title="Fall Detection Dashboard", layout="wide")
    st.title("Fall Detection Dashboard - Video Test")
    st.caption(
        "Dashboard nay giup kiem chung mo hinh tren video co san thay vi webcam: "
        "ban nhin duoc overlay skeleton, chi so dong hoc, va event log de danh gia do on dinh cua he thong."
    )

    if "event_log" not in st.session_state:
        st.session_state.event_log = []
    if "last_event" not in st.session_state:
        st.session_state.last_event = None
    if "last_feature_seq" not in st.session_state:
        st.session_state.last_feature_seq = None
    if "video_state" not in st.session_state:
        st.session_state.video_state = "idle"  # idle | playing | paused
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "worker" not in st.session_state:
        st.session_state.worker = None
    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None
    if "current_video_path" not in st.session_state:
        st.session_state.current_video_path = None
    if "vy_hist" not in st.session_state:
        st.session_state.vy_hist = []
    if "angle_hist" not in st.session_state:
        st.session_state.angle_hist = []
    if "frame_count" not in st.session_state:
        st.session_state.frame_count = 0
    if "t_start" not in st.session_state:
        st.session_state.t_start = 0.0
    if "last_out" not in st.session_state:
        st.session_state.last_out = None

    def _cleanup_runtime(clear_temp: bool = False):
        if st.session_state.worker is not None:
            st.session_state.worker.stop()
            st.session_state.worker = None
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        if clear_temp and st.session_state.temp_video_path:
            Path(st.session_state.temp_video_path).unlink(missing_ok=True)
            st.session_state.temp_video_path = None

    def _reset_to_waiting_state():
        _cleanup_runtime(clear_temp=True)
        st.session_state.video_state = "idle"
        st.session_state.vy_hist = []
        st.session_state.angle_hist = []
        st.session_state.frame_count = 0
        st.session_state.t_start = 0.0
        st.session_state.last_out = None

    with st.sidebar:
        st.header("Video Input")
        uploaded = st.file_uploader("Upload video (.mp4, .avi)", type=["mp4", "avi", "mov", "mkv"])
        sample_videos = _discover_test_videos()
        selected = st.selectbox(
            "Hoac chon video test tu data/raw",
            options=["(Khong chon)"] + [str(p.relative_to(ROOT_DIR)) for p in sample_videos],
        )

        st.header("Model")
        model_candidates = sorted(MODELS_DIR.glob("*.keras")) if MODELS_DIR.exists() else []
        model_path = st.selectbox(
            "Chon model .keras",
            options=[str(p) for p in model_candidates] if model_candidates else [""],
        )

        st.header("Telegram Test")
        use_telegram = st.checkbox("Gui canh bao Telegram", value=False)
        bot_token = st.text_input("TELEGRAM_BOT_TOKEN", value="", type="password")
        chat_id = st.text_input("TELEGRAM_CHAT_ID", value="", type="password")
        location = st.text_input("Vi tri/Phong", value="Home_01")

        run_btn = st.button("Tai video vao he thong", type="primary")

    # Load/replace source video + runtime resources
    if run_btn:
        if not model_path:
            st.error("Khong tim thay model .keras trong thu muc models/.")
            return

        video_path = None
        if uploaded is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
            temp_file.write(uploaded.read())
            temp_file.flush()
            video_path = temp_file.name
        elif selected != "(Khong chon)":
            video_path = str(ROOT_DIR / selected)
            temp_file = None
        else:
            st.error("Vui long upload hoac chon mot video.")
            return

        _reset_to_waiting_state()

        yolo_model = load_yolo()
        bilstm_model = load_bilstm(model_path)
        try:
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        except Exception as exc:
            st.warning(f"[WARN] Cannot open video {video_path}: {exc}")
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)
            return

        if not cap.isOpened():
            st.error("Khong mo duoc video.")
            if temp_file is not None:
                Path(temp_file.name).unlink(missing_ok=True)
            return

        worker = InferenceWorker(
            yolo_model=yolo_model,
            bilstm_model=bilstm_model,
            send_telegram=use_telegram,
            location=location,
            bot_token=bot_token,
            chat_id=chat_id,
        )
        worker.start()

        st.session_state.cap = cap
        st.session_state.worker = worker
        st.session_state.current_video_path = video_path
        st.session_state.video_state = "paused"
        st.session_state.frame_count = 0
        st.session_state.t_start = time.time()
        st.session_state.temp_video_path = temp_file.name if temp_file is not None else None

    left_col, right_col = st.columns([2, 1])
    with left_col:
        video_slot = st.empty()
        control_cols = st.columns([1, 1, 1])
        with control_cols[0]:
            play_btn = st.button("Play", use_container_width=True)
        with control_cols[1]:
            pause_btn = st.button("Pause", use_container_width=True)
        with control_cols[2]:
            stop_btn = st.button("Stop", use_container_width=True)

        st.markdown(f"**Trang thai video:** `{st.session_state.video_state}`")
        st.markdown("### Event Log")
        log_df_slot = st.empty()
        log_img_slot = st.empty()
        fp_btn = st.button("Xac nhan bao dong gia (False Positive)")
    with right_col:
        fps_metric = st.empty()
        ram_metric = st.empty()
        gpu_metric = st.empty()
        conf_metric = st.empty()
        vy_metric = st.empty()
        angle_metric = st.empty()
        chart_slot = st.empty()

    # Transport controls
    if play_btn and st.session_state.cap is not None and st.session_state.worker is not None:
        st.session_state.video_state = "playing"
    if pause_btn and st.session_state.video_state == "playing":
        st.session_state.video_state = "paused"
    if stop_btn:
        _reset_to_waiting_state()

    # Process exactly one frame per rerun when playing (non-freezing UI model)
    if (
        st.session_state.video_state == "playing"
        and st.session_state.cap is not None
        and st.session_state.worker is not None
    ):
        ret, frame = st.session_state.cap.read()
        if not ret:
            _reset_to_waiting_state()
        else:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.worker.submit({"frame": frame, "timestamp": ts})
            try:
                out = st.session_state.worker.out_q.get(timeout=0.35)
                st.session_state.last_out = out
                st.session_state.frame_count += 1
                st.session_state.vy_hist.append(out["vy"])
                st.session_state.angle_hist.append(out["body_angle"])
                st.session_state.vy_hist = st.session_state.vy_hist[-200:]
                st.session_state.angle_hist = st.session_state.angle_hist[-200:]

                if out["event"] is not None:
                    evt = out["event"]
                    st.session_state.last_event = evt
                    st.session_state.event_log.append(
                        {
                            "timestamp": evt["timestamp"],
                            "peak_acc": round(float(evt["peak_acc"]), 4),
                            "hips_var": round(float(evt["hips_var"]), 6),
                        }
                    )
                if out["feature_seq"] is not None:
                    st.session_state.last_feature_seq = out["feature_seq"]
            except queue.Empty:
                pass

    # Render current snapshot (works for playing, paused, and idle)
    out = st.session_state.last_out
    if out is not None:
        frame_rgb = cv2.cvtColor(out["frame_overlay"], cv2.COLOR_BGR2RGB)
        video_slot.image(frame_rgb, channels="RGB", use_container_width=True)
    else:
        video_slot.info("Khung video se hien thi tai day. Bam 'Tai video vao he thong' roi bam Play.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=st.session_state.vy_hist, mode="lines", name="Vertical Velocity (Vy)"))
    fig.add_trace(go.Scatter(y=st.session_state.angle_hist, mode="lines", name="Body Tilt Angle"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    chart_slot.plotly_chart(fig, use_container_width=True)

    elapsed = max(time.time() - max(st.session_state.t_start, 1e-6), 1e-6)
    fps_val = st.session_state.frame_count / elapsed if st.session_state.frame_count > 0 else 0.0
    conf_val = out["prob"] if out is not None else 0.0
    vy_val = out["vy"] if out is not None else 0.0
    angle_val = out["body_angle"] if out is not None else 0.0
    ram_avg = out["ram_mb_avg"] if out is not None else 0.0
    gpu_avg = out["gpu_util_avg"] if out is not None else 0.0
    fps_metric.metric("FPS", f"{fps_val:.2f}")
    ram_metric.metric("RAM avg (MB)", f"{ram_avg:.1f}")
    gpu_metric.metric("GPU util avg (%)", f"{gpu_avg:.1f}")
    conf_metric.metric("Confidence Score", f"{conf_val:.3f}")
    vy_metric.metric("Vertical Velocity", f"{vy_val:.4f}")
    angle_metric.metric("Body Tilt Angle", f"{angle_val:.4f}")

    if st.session_state.last_event is not None:
        log_img_slot.image(
            cv2.cvtColor(st.session_state.last_event["skeleton_img"], cv2.COLOR_BGR2RGB),
            caption=f"Skeleton alert @ {st.session_state.last_event['timestamp']}",
            use_container_width=True,
        )
    if st.session_state.event_log:
        log_df_slot.dataframe(st.session_state.event_log, use_container_width=True)
    else:
        log_df_slot.info("Chua co su kien nga duoc xac nhan.")

    if fp_btn and st.session_state.last_event is not None:
        _save_false_positive(st.session_state.last_event, st.session_state.last_feature_seq)
        st.success("Da luu False Positive vao data/false_positives/")

    # Auto-rerun only when playing so video advances; paused keeps current frame/charts.
    if st.session_state.video_state == "playing":
        time.sleep(0.01)
        st.rerun()


if __name__ == "__main__":
    main()
