import copy
import difflib
import glob
import json
import math
import os
import queue
import random
import shutil
import threading
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog

warnings.filterwarnings("ignore")

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

DB_PATH = os.path.join(DATA_DIR, "db")
CONFIG_PATH = os.path.join(CONFIG_DIR, "gesture_config.json")
CUSTOM_GESTURE_PATH = os.path.join(CONFIG_DIR, "custom_gestures.json")

YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet.onnx")
SFACE_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
GESTURE_MODEL = os.path.join(MODEL_DIR, "gesture_recognizer.task")
VOSK_MODEL_DIR = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
GESTURE_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)
VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

MODE_SEQUENCE = ["presentation", "video", "youtube"]

ACTION_LABELS = {
    "toggle_mode": "Cycle mode",
    "next_slide": "Next",
    "previous_slide": "Previous",
    "start_slideshow": "Start show",
    "exit_slideshow": "Exit",
    "play_pause": "Play/Pause",
    "volume_up": "Vol +",
    "volume_down": "Vol -",
    "mute_toggle": "Mute",
    "seek_forward": "Forward",
    "seek_backward": "Back",
    "speed_up": "Speed +",
    "speed_down": "Speed -",
}

GESTURE_LABELS = {
    "Closed_Fist": "Fist",
    "Open_Palm": "Palm",
    "Pointing_Up": "Point",
    "Thumb_Down": "Thumb down",
    "Thumb_Up": "Thumb up",
    "Victory": "Victory",
    "ILoveYou": "I love you",
    "Swipe_Right": "Swipe right",
    "Swipe_Left": "Swipe left",
}

LIVENESS_PROMPTS = {
    "head_left": "Turn head LEFT",
    "head_right": "Turn head RIGHT",
    "hand_thumb_up": "Show THUMB UP",
    "hand_thumb_down": "Show THUMB DOWN",
    "hand_victory": "Show VICTORY",
    "hand_fist": "Show FIST",
    "hand_point": "Point UP",
}

LIVENESS_HAND_GESTURES = {
    "hand_thumb_up": "Thumb_Up",
    "hand_thumb_down": "Thumb_Down",
    "hand_victory": "Victory",
    "hand_fist": "Closed_Fist",
    "hand_point": "Pointing_Up",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "camera": {
        "primary_index": 0,
        "fallback_index": 1,
        "width": 960,
        "height": 540,
        "fps": 30,
        "mirror": True,
    },
    "face": {
        "score_threshold": 0.6,
        "nms_threshold": 0.3,
        "top_k": 5000,
        "track_iou_threshold": 0.25,
        "stable_seconds": 0.25,
        "unknown_retry_seconds": 2.0,
        "active_grace_seconds": 1.25,
        "sface_cosine_threshold": 0.363,
        "ambiguous_rank_gap": 0.15,
        "foreground_override_ratio": 1.12,
    },
    "emotion": {
        "enabled": True,
        "min_interval_seconds": 1.5,
    },
    "liveness": {
        "enabled": True,
        "steps_per_challenge": 2,
        "require_mixed_types": True,
        "step_timeout_seconds": 7.0,
        "success_hold_seconds": 0.35,
        "head_offset_threshold": 0.08,
        "hand_confidence_threshold": 0.55,
        "challenge_pool": [
            "head_left",
            "head_right",
            "hand_thumb_up",
            "hand_thumb_down",
            "hand_victory",
            "hand_fist",
            "hand_point",
        ],
    },
    "gestures": {
        "enabled": True,
        "confidence_threshold": 0.55,
        "stable_seconds": 0.35,
        "action_cooldown_seconds": 1.0,
        "custom_match_threshold": 0.20,
        "custom_capture_samples": 5,
        "custom_capture_interval_seconds": 0.45,
        "swipe_min_dx": 0.16,
        "swipe_max_dy": 0.14,
        "swipe_window_seconds": 0.75,
        "swipe_min_duration_seconds": 0.12,
        "swipe_hold_seconds": 0.55,
    },
    "ui": {
        "show_legend": True,
        "show_performance": True,
        "overlay_alpha": 0.68,
        "legend_position": "auto",
        "performance_position": "bottom_left",
    },
    "mode": "presentation",
    "bindings": {
        "global": {
            "ILoveYou": "toggle_mode",
        },
        "presentation": {
            "Thumb_Up": "next_slide",
            "Thumb_Down": "previous_slide",
            "Victory": "start_slideshow",
            "Closed_Fist": "exit_slideshow",
        },
        "video": {
            "Open_Palm": "play_pause",
            "Thumb_Up": "volume_up",
            "Thumb_Down": "volume_down",
            "Victory": "speed_up",
            "Closed_Fist": "speed_down",
            "Pointing_Up": "mute_toggle",
            "Swipe_Right": "seek_forward",
            "Swipe_Left": "seek_backward",
        },
        "youtube": {
            "Open_Palm": "play_pause",
            "Thumb_Up": "volume_up",
            "Thumb_Down": "volume_down",
            "Victory": "speed_up",
            "Closed_Fist": "speed_down",
            "Pointing_Up": "mute_toggle",
            "Swipe_Right": "seek_forward",
            "Swipe_Left": "seek_backward",
        },
    },
    "external_controls": {
        "enabled": True,
        "dry_run": False,
        "focus_before_action": True,
        "require_target_window": False,
        "active_profiles": {
            "presentation": "powerpoint",
            "video": "generic_video",
            "youtube": "youtube_video",
        },
        "target_windows": {
            "powerpoint": ["PowerPoint", "Slide Show"],
            "generic_video": ["VLC", "Media Player", "Movies & TV", "Films & TV"],
            "youtube_video": ["YouTube", "Chrome", "Microsoft Edge", "Firefox"],
        },
        "profiles": {
            "powerpoint": {
                "next_slide": ["right"],
                "previous_slide": ["left"],
                "start_slideshow": ["f5"],
                "exit_slideshow": ["esc"],
            },
            "generic_video": {
                "play_pause": ["space"],
                "volume_up": ["volumeup"],
                "volume_down": ["volumedown"],
                "mute_toggle": ["volumemute"],
                "seek_forward": ["right"],
                "seek_backward": ["left"],
                "speed_up": ["]"],
                "speed_down": ["["],
            },
            "youtube_video": {
                "play_pause": ["space"],
                "volume_up": ["up"],
                "volume_down": ["down"],
                "mute_toggle": ["m"],
                "seek_forward": ["right"],
                "seek_backward": ["left"],
                "speed_up": ["shift", "."],
                "speed_down": ["shift", ","],
            },
        },
    },
    "voice": {
        "enabled": False,
        "model_path": "models/vosk-model-small-en-us-0.15",
        "model_url": VOSK_MODEL_URL,
        "device": None,
        "sample_rate": 16000,
        "blocksize": 8000,
        "command_cooldown_seconds": 1.0,
        "max_command_age_seconds": 2.0,
        "phrase_match_threshold": 0.85,
        "legend_max_phrases": 9,
        "bindings": {
            "global": {
                "cycle mode": "toggle_mode",
                "change mode": "toggle_mode",
                "switch mode": "toggle_mode",
            },
            "presentation": {
                "next slide": "next_slide",
                "next": "next_slide",
                "previous slide": "previous_slide",
                "previous": "previous_slide",
                "start slideshow": "start_slideshow",
                "start show": "start_slideshow",
                "exit slideshow": "exit_slideshow",
                "exit show": "exit_slideshow",
            },
            "video": {
                "play pause": "play_pause",
                "pause": "play_pause",
                "play": "play_pause",
                "mute": "mute_toggle",
                "unmute": "mute_toggle",
                "volume up": "volume_up",
                "volume down": "volume_down",
                "speed up": "speed_up",
                "speed down": "speed_down",
                "seek forward": "seek_forward",
                "skip forward": "seek_forward",
                "seek backward": "seek_backward",
                "skip backward": "seek_backward",
            },
            "youtube": {
                "play pause": "play_pause",
                "pause": "play_pause",
                "play": "play_pause",
                "mute": "mute_toggle",
                "unmute": "mute_toggle",
                "volume up": "volume_up",
                "volume down": "volume_down",
                "speed up": "speed_up",
                "speed down": "speed_down",
                "seek forward": "seek_forward",
                "skip forward": "seek_forward",
                "seek backward": "seek_backward",
                "skip backward": "seek_backward",
            },
        },
    },
}


@dataclass
class FaceEntry:
    person: str
    path: str
    feature: np.ndarray


@dataclass
class FaceTrack:
    track_id: int
    face: np.ndarray
    bbox: Tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    identity: str = "Unknown"
    similarity: float = 0.0
    identified_at: float = 0.0
    last_attempt: float = 0.0
    emotion: str = "N/A"
    emotion_at: float = 0.0
    seen_count: int = 1
    pending_identification: bool = True


@dataclass
class GestureFrame:
    name: str = ""
    score: float = 0.0
    handedness: str = ""
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    timestamp_ms: int = 0


@dataclass
class VoiceCommand:
    text: str = ""
    timestamp: float = 0.0


@dataclass
class PerformanceStats:
    fps: float = 0.0
    frame_count: int = 0
    last_fps_at: float = 0.0
    detect_ms: float = 0.0
    tracking_ms: float = 0.0
    gesture_latency_ms: float = 0.0
    action_ms: float = 0.0

    def tick(self, now: float) -> None:
        self.frame_count += 1
        if self.last_fps_at == 0.0:
            self.last_fps_at = now
            return
        elapsed = now - self.last_fps_at
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_at = now

    def observe(self, field_name: str, value: float) -> None:
        previous = getattr(self, field_name)
        if previous == 0.0:
            setattr(self, field_name, value)
        else:
            setattr(self, field_name, previous * 0.85 + value * 0.15)


def deep_merge(defaults: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(path: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(path):
        save_json(path, defaults)
        return copy.deepcopy(defaults)

    try:
        with open(path, "r", encoding="utf-8") as f:
            current = json.load(f)
    except (OSError, json.JSONDecodeError):
        backup = f"{path}.broken"
        try:
            os.replace(path, backup)
            print(f"Config was invalid. Moved it to {backup}")
        except OSError:
            pass
        save_json(path, defaults)
        return copy.deepcopy(defaults)

    merged = deep_merge(defaults, current)
    if merged != current:
        save_json(path, merged)
    return merged


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def ensure_file(path: str, url: str, label: str) -> bool:
    if os.path.exists(path):
        return True

    print(f"Downloading {label}...")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {label}: {path}")
        return True
    except Exception as exc:
        print(f"Could not download {label}: {exc}")
        return False


def resolve_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def ensure_vosk_model(config: Dict[str, Any]) -> bool:
    voice_cfg = config.get("voice", {})
    model_path = resolve_project_path(str(voice_cfg.get("model_path", VOSK_MODEL_DIR)))
    if os.path.isdir(model_path):
        return True

    url = str(voice_cfg.get("model_url", VOSK_MODEL_URL))
    if not url:
        print(f"Voice model missing: {model_path}")
        return False

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    archive_path = f"{model_path}.zip"
    print(f"Downloading Vosk voice model...")
    try:
        urllib.request.urlretrieve(url, archive_path)
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(os.path.dirname(model_path))
        if os.path.exists(archive_path):
            os.remove(archive_path)
        if os.path.isdir(model_path):
            print(f"Voice model ready: {model_path}")
            return True
        print(f"Voice model archive did not create expected folder: {model_path}")
        return False
    except Exception as exc:
        print(f"Could not prepare Vosk voice model: {exc}")
        try:
            if os.path.exists(archive_path):
                os.remove(archive_path)
        except OSError:
            pass
        return False


def normalize_voice_phrase(text: str) -> str:
    lowered = text.lower().replace("_", " ")
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in lowered)
    return " ".join(cleaned.split())


def voice_bindings_for_mode(config: Dict[str, Any], mode: str) -> Dict[str, str]:
    voice_cfg = config.get("voice", {})
    raw_bindings = voice_cfg.get("bindings", {})
    bindings: Dict[str, str] = {}
    for section in ("global", mode):
        section_bindings = raw_bindings.get(section, {})
        if not isinstance(section_bindings, dict):
            continue
        for phrase, action in section_bindings.items():
            normalized = normalize_voice_phrase(str(phrase))
            if normalized and normalized not in bindings:
                bindings[normalized] = str(action)
    return bindings


def all_voice_phrases(config: Dict[str, Any]) -> List[str]:
    voice_cfg = config.get("voice", {})
    raw_bindings = voice_cfg.get("bindings", {})
    phrases: List[str] = []
    seen: set[str] = set()
    if isinstance(raw_bindings, dict):
        for section_bindings in raw_bindings.values():
            if not isinstance(section_bindings, dict):
                continue
            for phrase in section_bindings.keys():
                normalized = normalize_voice_phrase(str(phrase))
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    phrases.append(normalized)
    return phrases


def match_voice_action(config: Dict[str, Any], mode: str, text: str) -> Tuple[str, float, str]:
    normalized = normalize_voice_phrase(text)
    if not normalized:
        return "", 0.0, ""

    bindings = voice_bindings_for_mode(config, mode)
    if normalized in bindings:
        return bindings[normalized], 1.0, normalized

    best_phrase = ""
    best_score = 0.0
    for phrase in bindings.keys():
        score = difflib.SequenceMatcher(None, normalized, phrase).ratio()
        if score > best_score:
            best_score = score
            best_phrase = phrase

    threshold = float(config.get("voice", {}).get("phrase_match_threshold", 0.85))
    if best_phrase and best_score >= threshold:
        return bindings[best_phrase], best_score, best_phrase
    return "", best_score, best_phrase


def check_db(db_path: str) -> bool:
    for root, _, files in os.walk(db_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
    return False


def image_paths(db_path: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(db_path):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def create_detector(config: Dict[str, Any]) -> Optional[Any]:
    face_cfg = config["face"]
    try:
        return cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=(320, 320),
            score_threshold=float(face_cfg["score_threshold"]),
            nms_threshold=float(face_cfg["nms_threshold"]),
            top_k=int(face_cfg["top_k"]),
        )
    except Exception as exc:
        print(f"Error loading YuNet model: {exc}")
        return None


def create_recognizer() -> Optional[Any]:
    try:
        return cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
    except Exception as exc:
        print(f"Error loading SFace model: {exc}")
        return None


def detect_faces(detector: Any, frame: np.ndarray) -> List[np.ndarray]:
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    return sorted([face for face in faces], key=lambda f: f[2] * f[3], reverse=True)


def face_bbox(face: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, y, box_w, box_h = map(int, face[:4])
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + box_w)
    y2 = min(h, y + box_h)
    return x1, y1, x2, y2


def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def crop_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return None
    return crop.copy()


def face_landmarks(face: np.ndarray) -> List[Tuple[float, float]]:
    if len(face) < 15:
        return []
    values = face[4:14]
    return [
        (float(values[i]), float(values[i + 1]))
        for i in range(0, len(values), 2)
    ]


def nose_offset_ratio(track: FaceTrack) -> Optional[float]:
    landmarks = face_landmarks(track.face)
    if len(landmarks) < 3:
        return None
    nose_x, _ = landmarks[2]
    x1, _, x2, _ = track.bbox
    width = max(1.0, float(x2 - x1))
    center_x = (x1 + x2) / 2.0
    return (nose_x - center_x) / width


class FaceIndex:
    def __init__(
        self,
        db_path: str,
        detector: Any,
        recognizer: Any,
        threshold: float,
    ) -> None:
        self.db_path = db_path
        self.detector = detector
        self.recognizer = recognizer
        self.threshold = threshold
        self.entries: List[FaceEntry] = []

    def reload(self) -> None:
        self.entries.clear()
        os.makedirs(self.db_path, exist_ok=True)
        for path in image_paths(self.db_path):
            person = os.path.basename(os.path.dirname(path))
            img = cv2.imread(path)
            if img is None:
                continue
            feature = self._feature_from_database_image(img)
            if feature is not None:
                self.entries.append(FaceEntry(person=person, path=path, feature=feature))

        people = sorted({entry.person for entry in self.entries})
        print(f"SFace identity index: {len(self.entries)} samples, {len(people)} people.")

    def _feature_from_database_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            faces = detect_faces(self.detector, img)
            if faces:
                aligned = self.recognizer.alignCrop(img, faces[0])
            else:
                aligned = cv2.resize(img, (112, 112))
            feature = self.recognizer.feature(aligned)
            return np.asarray(feature).copy()
        except Exception:
            return None

    def identify(self, frame: np.ndarray, face: np.ndarray) -> Tuple[str, float]:
        if not self.entries:
            return "Unknown", 0.0
        try:
            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned)
        except Exception:
            return "Unknown", 0.0

        best_person = "Unknown"
        best_score = -1.0
        for entry in self.entries:
            try:
                score = float(
                    self.recognizer.match(
                        feature,
                        entry.feature,
                        cv2.FaceRecognizerSF_FR_COSINE,
                    )
                )
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_person = entry.person

        if best_score >= self.threshold:
            return best_person, best_score
        return "Unknown", max(0.0, best_score)


class FaceTracker:
    def __init__(self, config: Dict[str, Any], face_index: FaceIndex) -> None:
        self.config = config
        self.face_index = face_index
        self.tracks: Dict[int, FaceTrack] = {}
        self.next_id = 1
        self.active_track_id: Optional[int] = None
        self.lock_reason = "Waiting for authorized user"

    def reset_identities(self) -> None:
        for track in self.tracks.values():
            track.identity = "Unknown"
            track.similarity = 0.0
            track.pending_identification = True
            track.last_attempt = 0.0
        self.active_track_id = None
        self.lock_reason = "Identity database changed"

    def update(self, frame: np.ndarray, faces: List[np.ndarray], now: float) -> List[FaceTrack]:
        assigned_tracks: set[int] = set()
        assigned_faces: set[int] = set()
        face_cfg = self.config["face"]
        iou_threshold = float(face_cfg["track_iou_threshold"])

        visible_bboxes = [face_bbox(face, frame.shape) for face in faces]
        for face_idx, bbox in enumerate(visible_bboxes):
            best_track_id = None
            best_iou = 0.0
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks:
                    continue
                score = iou(track.bbox, bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= iou_threshold:
                track = self.tracks[best_track_id]
                track.face = faces[face_idx]
                track.bbox = bbox
                track.last_seen = now
                track.seen_count += 1
                assigned_tracks.add(best_track_id)
                assigned_faces.add(face_idx)

        for face_idx, face in enumerate(faces):
            if face_idx in assigned_faces:
                continue
            bbox = visible_bboxes[face_idx]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = FaceTrack(
                track_id=track_id,
                face=face,
                bbox=bbox,
                first_seen=now,
                last_seen=now,
            )

        self._drop_stale(now)
        self._identify_ready_tracks(frame, now)
        self._select_active(frame.shape, now)
        return self.visible_tracks(now)

    def visible_tracks(self, now: float) -> List[FaceTrack]:
        return [
            track
            for track in self.tracks.values()
            if now - track.last_seen <= 0.35
        ]

    def active_track(self, now: float) -> Optional[FaceTrack]:
        if self.active_track_id is None:
            return None
        track = self.tracks.get(self.active_track_id)
        if track is None:
            return None
        if now - track.last_seen <= float(self.config["face"]["active_grace_seconds"]):
            return track
        return None

    def is_unlocked(self, now: float) -> bool:
        track = self.active_track(now)
        return track is not None and track.identity != "Unknown"

    def _drop_stale(self, now: float) -> None:
        grace = float(self.config["face"]["active_grace_seconds"])
        stale_ids = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track.last_seen > max(3.0, grace + 1.0)
        ]
        for track_id in stale_ids:
            self.tracks.pop(track_id, None)
            if self.active_track_id == track_id:
                self.active_track_id = None

    def _identify_ready_tracks(self, frame: np.ndarray, now: float) -> None:
        face_cfg = self.config["face"]
        stable_seconds = float(face_cfg["stable_seconds"])
        unknown_retry = float(face_cfg["unknown_retry_seconds"])

        for track in self.visible_tracks(now):
            if now - track.first_seen < stable_seconds:
                continue
            if track.pending_identification:
                self._identify_track(frame, track, now)
            elif track.identity == "Unknown" and now - track.last_attempt >= unknown_retry:
                self._identify_track(frame, track, now)

    def _identify_track(self, frame: np.ndarray, track: FaceTrack, now: float) -> None:
        identity, score = self.face_index.identify(frame, track.face)
        track.identity = identity
        track.similarity = score
        track.identified_at = now
        track.last_attempt = now
        track.pending_identification = False

    def _select_active(self, frame_shape: Tuple[int, int, int], now: float) -> None:
        visible = self.visible_tracks(now)
        authorized = [track for track in visible if track.identity != "Unknown"]
        if not visible:
            if self.active_track(now) is None:
                self.active_track_id = None
                self.lock_reason = "No visible face"
            return

        if not authorized:
            self.active_track_id = None
            self.lock_reason = "No authorized face"
            return

        ranked = sorted(
            authorized,
            key=lambda track: self._controller_rank(track, frame_shape),
            reverse=True,
        )
        best = ranked[0]
        best_rank = self._controller_rank(best, frame_shape)

        if len(ranked) > 1:
            second_rank = self._controller_rank(ranked[1], frame_shape)
            if best_rank - second_rank < float(self.config["face"]["ambiguous_rank_gap"]):
                self.active_track_id = None
                self.lock_reason = "Ambiguous authorized controller"
                return

        largest = max(visible, key=lambda track: bbox_area(track.bbox))
        if largest.identity == "Unknown":
            ratio = float(self.config["face"]["foreground_override_ratio"])
            if bbox_area(largest.bbox) > bbox_area(best.bbox) * ratio:
                self.active_track_id = None
                self.lock_reason = "Foreground face is not authorized"
                return

        self.active_track_id = best.track_id
        self.lock_reason = f"Unlocked: {best.identity}"

    def _controller_rank(self, track: FaceTrack, frame_shape: Tuple[int, int, int]) -> float:
        h, w = frame_shape[:2]
        area_score = bbox_area(track.bbox) / max(1.0, float(w * h))
        cx, cy = bbox_center(track.bbox)
        dx = abs(cx - (w / 2.0)) / max(1.0, w / 2.0)
        dy = abs(cy - (h / 2.0)) / max(1.0, h / 2.0)
        center_score = 1.0 - min(1.0, math.sqrt(dx * dx + dy * dy) / math.sqrt(2.0))
        return area_score * 4.0 + center_score

    def hand_belongs_to_active(
        self,
        landmarks: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, int, int],
        now: float,
    ) -> bool:
        active = self.active_track(now)
        if active is None or not landmarks:
            return False

        visible = self.visible_tracks(now)
        if len(visible) <= 1:
            return True

        h, w = frame_shape[:2]
        hand_x = float(np.mean([point[0] for point in landmarks])) * w
        hand_y = float(np.mean([point[1] for point in landmarks])) * h

        def distance_to_track(track: FaceTrack) -> float:
            cx, cy = bbox_center(track.bbox)
            width = max(1.0, track.bbox[2] - track.bbox[0])
            height = max(1.0, track.bbox[3] - track.bbox[1])
            return math.sqrt(((hand_x - cx) / width) ** 2 + ((hand_y - cy) / height) ** 2)

        nearest = min(visible, key=distance_to_track)
        return nearest.track_id == active.track_id


class EmotionWorker:
    def __init__(self, enabled: bool, min_interval: float) -> None:
        self.enabled = enabled
        self.min_interval = min_interval
        self.tasks: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(maxsize=1)
        self.results: Dict[int, Tuple[str, float]] = {}
        self.inflight: set[int] = set()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        if self.enabled:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def submit(self, track: FaceTrack, crop: Optional[np.ndarray], now: float) -> None:
        if not self.enabled or crop is None:
            return
        if now - track.emotion_at < self.min_interval:
            return
        with self.lock:
            if track.track_id in self.inflight:
                return
            self.inflight.add(track.track_id)
        try:
            self.tasks.put_nowait((track.track_id, crop))
            track.emotion_at = now
        except queue.Full:
            with self.lock:
                self.inflight.discard(track.track_id)

    def latest(self, track_id: int) -> str:
        with self.lock:
            result = self.results.get(track_id)
        return result[0] if result else "N/A"

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        try:
            from deepface import DeepFace
        except Exception as exc:
            print(f"DeepFace emotion worker disabled: {exc}")
            self.enabled = False
            return

        while not self.stop_event.is_set():
            try:
                track_id, crop = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue

            emotion = "N/A"
            try:
                result = DeepFace.analyze(
                    crop,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip",
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]
                emotion = str(result.get("dominant_emotion", "N/A"))
            except Exception:
                pass

            with self.lock:
                self.results[track_id] = (emotion, time.time())
                self.inflight.discard(track_id)


class CustomGestureStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            payload = {"templates": []}
            save_json(self.path, payload)
            return payload
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {"templates": []}
        if "templates" not in data or not isinstance(data["templates"], list):
            data["templates"] = []
        return data

    def save(self) -> None:
        save_json(self.path, self.data)

    def add_template(
        self,
        name: str,
        action: str,
        samples: List[List[float]],
    ) -> None:
        self.data["templates"] = [
            item for item in self.data["templates"] if item.get("name") != name
        ]
        self.data["templates"].append(
            {
                "name": name,
                "action": action,
                "samples": samples,
            }
        )
        self.save()

    def match(
        self,
        landmarks: List[Tuple[float, float, float]],
        threshold: float,
    ) -> Optional[Tuple[str, str, float]]:
        vector = normalize_landmarks(landmarks)
        if vector is None:
            return None

        best: Optional[Tuple[str, str, float]] = None
        for template in self.data.get("templates", []):
            samples = template.get("samples", [])
            if not samples:
                continue
            distances = [
                landmark_distance(vector, sample)
                for sample in samples
                if isinstance(sample, list) and len(sample) == len(vector)
            ]
            if not distances:
                continue
            distance = min(distances)
            if best is None or distance < best[2]:
                best = (
                    str(template.get("name", "custom")),
                    str(template.get("action", "")),
                    distance,
                )

        if best and best[2] <= threshold:
            return best
        return None


def normalize_landmarks(
    landmarks: List[Tuple[float, float, float]]
) -> Optional[List[float]]:
    if len(landmarks) < 21:
        return None
    wrist = np.array(landmarks[0], dtype=np.float32)
    points = np.array(landmarks, dtype=np.float32) - wrist
    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale < 1e-6:
        return None
    points = points / scale
    return points.reshape(-1).astype(float).tolist()


def landmark_distance(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)))


class GestureCaptureState:
    def __init__(self) -> None:
        self.active = False
        self.name = ""
        self.action = ""
        self.samples: List[List[float]] = []
        self.last_sample_at = 0.0

    def start(self, name: str, action: str) -> None:
        self.active = True
        self.name = name
        self.action = action
        self.samples = []
        self.last_sample_at = 0.0

    def reset(self) -> None:
        self.active = False
        self.name = ""
        self.action = ""
        self.samples = []
        self.last_sample_at = 0.0


class MediaPipeGestureEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.enabled = bool(config["gestures"]["enabled"])
        self.last_result = GestureFrame()
        self.lock = threading.Lock()
        self.recognizer = None
        self.mp = None

        if not self.enabled:
            return
        try:
            import mediapipe as mp

            self.mp = mp
            BaseOptions = mp.tasks.BaseOptions
            GestureRecognizer = mp.tasks.vision.GestureRecognizer
            GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=GESTURE_MODEL),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._callback,
            )
            self.recognizer = GestureRecognizer.create_from_options(options)
        except Exception as exc:
            self.enabled = False
            print(f"Gesture control disabled. Install mediapipe and verify model file. Details: {exc}")

    def close(self) -> None:
        if self.recognizer is not None:
            self.recognizer.close()

    def submit(self, frame: np.ndarray, timestamp_ms: int) -> None:
        if not self.enabled or self.recognizer is None or self.mp is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            self.recognizer.recognize_async(mp_image, timestamp_ms)
        except Exception:
            pass

    def latest(self) -> GestureFrame:
        with self.lock:
            return copy.deepcopy(self.last_result)

    def _callback(self, result: Any, output_image: Any, timestamp_ms: int) -> None:
        frame = GestureFrame(timestamp_ms=timestamp_ms)
        try:
            if result.gestures and result.gestures[0]:
                category = result.gestures[0][0]
                frame.name = category.category_name
                frame.score = float(category.score)
            if result.handedness and result.handedness[0]:
                frame.handedness = result.handedness[0][0].category_name
            if result.hand_landmarks and result.hand_landmarks[0]:
                frame.landmarks = [
                    (float(point.x), float(point.y), float(point.z))
                    for point in result.hand_landmarks[0]
                ]
        except Exception:
            frame = GestureFrame(timestamp_ms=timestamp_ms)

        with self.lock:
            self.last_result = frame


class MotionGestureDetector:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.history: List[Tuple[float, float, float]] = []
        self.active_name = ""
        self.active_until = 0.0

    def process(
        self,
        landmarks: List[Tuple[float, float, float]],
        authorized: bool,
        hand_ok: bool,
        now: float,
    ) -> Optional[GestureFrame]:
        if not authorized or not hand_ok or not landmarks:
            self.history.clear()
            self.active_name = ""
            self.active_until = 0.0
            return None

        if self.active_name and now <= self.active_until:
            return GestureFrame(name=self.active_name, score=1.0, timestamp_ms=int(now * 1000))
        if self.active_name and now > self.active_until:
            self.active_name = ""

        x = float(np.mean([point[0] for point in landmarks]))
        y = float(np.mean([point[1] for point in landmarks]))
        self.history.append((now, x, y))

        gesture_cfg = self.config["gestures"]
        window = float(gesture_cfg["swipe_window_seconds"])
        self.history = [item for item in self.history if now - item[0] <= window]
        if len(self.history) < 2:
            return None

        start_t, start_x, start_y = self.history[0]
        end_t, end_x, end_y = self.history[-1]
        duration = end_t - start_t
        dx = end_x - start_x
        dy = end_y - start_y

        if duration < float(gesture_cfg["swipe_min_duration_seconds"]):
            return None
        if abs(dx) < float(gesture_cfg["swipe_min_dx"]):
            return None
        if abs(dy) > float(gesture_cfg["swipe_max_dy"]):
            return None

        self.active_name = "Swipe_Right" if dx > 0 else "Swipe_Left"
        self.active_until = now + float(gesture_cfg["swipe_hold_seconds"])
        self.history.clear()
        return GestureFrame(name=self.active_name, score=1.0, timestamp_ms=int(now * 1000))


class ActionController:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.mode = str(config.get("mode", "presentation"))
        if self.mode not in MODE_SEQUENCE:
            self.mode = "presentation"
            self.config["mode"] = self.mode
        self.enabled = bool(config["external_controls"]["enabled"])
        self.pyautogui = None
        self.pygetwindow = None
        self.last_focus_status = ""
        if not self.enabled:
            return
        try:
            import pyautogui

            pyautogui.PAUSE = 0.03
            self.pyautogui = pyautogui
            try:
                import pygetwindow

                self.pygetwindow = pygetwindow
            except Exception:
                self.pygetwindow = None
        except Exception as exc:
            self.enabled = False
            print(f"External key controls disabled. Install pyautogui. Details: {exc}")

    def set_mode(self, mode: str) -> None:
        if mode in MODE_SEQUENCE:
            self.mode = mode
            self.config["mode"] = mode
            save_json(CONFIG_PATH, self.config)

    def toggle_mode(self) -> str:
        current_index = MODE_SEQUENCE.index(self.mode) if self.mode in MODE_SEQUENCE else 0
        self.set_mode(MODE_SEQUENCE[(current_index + 1) % len(MODE_SEQUENCE)])
        return self.mode

    def toggle_dry_run(self) -> bool:
        controls = self.config["external_controls"]
        controls["dry_run"] = not bool(controls.get("dry_run", False))
        save_json(CONFIG_PATH, self.config)
        return bool(controls["dry_run"])

    def run_action(self, action: str) -> str:
        if action == "toggle_mode":
            mode = self.toggle_mode()
            return f"Mode: {mode}"

        if not self.enabled or self.pyautogui is None:
            return "External controls disabled"

        profile_name = self.config["external_controls"]["active_profiles"].get(self.mode, "")
        profile = self.config["external_controls"]["profiles"].get(profile_name, {})
        keys = profile.get(action)
        if not keys:
            return f"No key binding for {action}"

        if bool(self.config["external_controls"].get("dry_run", False)):
            return f"DRY RUN {self.mode}: {action} -> {'+'.join(keys)}"

        try:
            focus_status = self._focus_target_window(profile_name)
            if (
                focus_status.startswith("Target not found")
                and bool(self.config["external_controls"].get("require_target_window", False))
            ):
                return focus_status
            if len(keys) == 1:
                self.pyautogui.press(keys[0])
            else:
                self.pyautogui.hotkey(*keys)
            suffix = f" | {focus_status}" if focus_status else ""
            return f"{self.mode}: {action}{suffix}"
        except Exception as exc:
            return f"Action failed: {exc}"

    def valid_actions(self) -> List[str]:
        actions = {"toggle_mode"}
        profiles = self.config["external_controls"].get("profiles", {})
        for profile in profiles.values():
            actions.update(profile.keys())
        return sorted(actions)

    def _focus_target_window(self, profile_name: str) -> str:
        controls = self.config["external_controls"]
        if not bool(controls.get("focus_before_action", True)):
            self.last_focus_status = "Focus disabled"
            return ""
        if self.pygetwindow is None:
            self.last_focus_status = "Window focus unavailable"
            return self.last_focus_status

        titles = controls.get("target_windows", {}).get(profile_name, [])
        if not titles:
            self.last_focus_status = "No target title configured"
            return self.last_focus_status

        for title in titles:
            try:
                windows = self.pygetwindow.getWindowsWithTitle(title)
            except Exception:
                windows = []
            for window in windows:
                try:
                    if getattr(window, "isMinimized", False):
                        window.restore()
                    window.activate()
                    time.sleep(0.06)
                    self.last_focus_status = f"Focused: {title}"
                    return self.last_focus_status
                except Exception:
                    continue

        self.last_focus_status = f"Target not found: {profile_name}"
        return self.last_focus_status


class VoiceCommandEngine:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.command_queue: queue.Queue[VoiceCommand] = queue.Queue()
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.sd = None
        self.model = None
        self.recognizer_cls = None
        self.listening = False
        self.status = "Voice off"
        self.last_text = ""
        self.last_action = ""
        self.last_heard_at = 0.0
        self.last_action_at: Dict[str, float] = {}
        if bool(self.config.get("voice", {}).get("enabled", False)):
            self.start(persist=False)

    def toggle(self) -> str:
        voice_cfg = self.config.setdefault("voice", {})
        if self.listening or bool(voice_cfg.get("enabled", False)):
            self.stop(persist=True)
            return self.status_text()

        status = self.start(persist=False)
        if self.listening:
            voice_cfg["enabled"] = True
            save_json(CONFIG_PATH, self.config)
        return status

    def start(self, persist: bool = False) -> str:
        if self.listening:
            return self.status_text()
        if not self._load_backend():
            return self.status_text()

        if persist:
            self.config.setdefault("voice", {})["enabled"] = True
            save_json(CONFIG_PATH, self.config)

        self.stop_event.clear()
        self.audio_queue = queue.Queue()
        self.listening = True
        self.thread = threading.Thread(target=self._run, name="voice-listener", daemon=True)
        self.thread.start()
        self._set_status("Voice starting")
        return self.status_text()

    def stop(self, persist: bool = False) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.listening = False
        self.clear_commands()
        if persist:
            self.config.setdefault("voice", {})["enabled"] = False
            save_json(CONFIG_PATH, self.config)
        self._set_status("Voice off")

    def close(self) -> None:
        self.stop(persist=False)

    def clear_commands(self) -> None:
        while True:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                return

    def pop_latest(self) -> Optional[VoiceCommand]:
        latest = None
        while True:
            try:
                latest = self.command_queue.get_nowait()
            except queue.Empty:
                return latest

    def process_latest(
        self,
        actions: ActionController,
        authorized: bool,
        capture_active: bool,
        now: float,
        stats: Optional[PerformanceStats] = None,
    ) -> Tuple[str, str]:
        if not self.should_show():
            return "", ""
        if not authorized or capture_active:
            self.clear_commands()
            return "", ""

        command = self.pop_latest()
        if command is None:
            return "", ""

        voice_cfg = self.config.get("voice", {})
        max_age = float(voice_cfg.get("max_command_age_seconds", 2.0))
        if now - command.timestamp > max_age:
            return "Voice: ignored stale command", ""

        action, score, matched_phrase = match_voice_action(self.config, actions.mode, command.text)
        with self.lock:
            self.last_text = command.text
            self.last_heard_at = command.timestamp
        if not action:
            detail = f" near '{matched_phrase}'" if matched_phrase else ""
            return f"Voice heard: {command.text}{detail}", ""

        cooldown = float(voice_cfg.get("command_cooldown_seconds", 1.0))
        last_at = self.last_action_at.get(action, 0.0)
        if now - last_at < cooldown:
            return "", ""

        self.last_action_at[action] = now
        with self.lock:
            self.last_action = action
        action_start = time.perf_counter()
        status = actions.run_action(action)
        if stats is not None:
            stats.observe("action_ms", (time.perf_counter() - action_start) * 1000.0)
        phrase = matched_phrase or command.text
        return f"Voice: {phrase} ({score:.2f}) -> {status}", action

    def should_show(self) -> bool:
        return bool(self.config.get("voice", {}).get("enabled", False)) or self.listening

    def status_text(self) -> str:
        with self.lock:
            return self.status

    def snapshot(self) -> Tuple[str, str, str]:
        with self.lock:
            return self.status, self.last_text, self.last_action

    def _load_backend(self) -> bool:
        if self.sd is not None and self.model is not None and self.recognizer_cls is not None:
            return True
        try:
            import sounddevice as sd
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            if not ensure_vosk_model(self.config):
                self._set_status("Voice unavailable: missing Vosk model")
                return False
            model_path = resolve_project_path(str(self.config["voice"]["model_path"]))
            self.sd = sd
            self.recognizer_cls = KaldiRecognizer
            self.model = Model(model_path)
            return True
        except Exception as exc:
            self._set_status(f"Voice unavailable: {exc}")
            return False

    def _run(self) -> None:
        assert self.sd is not None
        assert self.model is not None
        assert self.recognizer_cls is not None

        voice_cfg = self.config.get("voice", {})
        sample_rate = int(voice_cfg.get("sample_rate", 16000))
        blocksize = int(voice_cfg.get("blocksize", 8000))
        device = voice_cfg.get("device", None)
        if device == "":
            device = None

        phrases = all_voice_phrases(self.config)
        grammar = json.dumps(phrases + ["[unk]"])
        recognizer = self.recognizer_cls(self.model, sample_rate, grammar)

        def callback(indata: bytes, frames: int, callback_time: Any, status: Any) -> None:
            if status:
                self._set_status(f"Voice warning: {status}")
            if not self.stop_event.is_set():
                self.audio_queue.put(bytes(indata))

        try:
            with self.sd.RawInputStream(
                samplerate=sample_rate,
                blocksize=blocksize,
                device=device,
                dtype="int16",
                channels=1,
                callback=callback,
            ):
                self.listening = True
                self._set_status("Voice listening")
                while not self.stop_event.is_set():
                    try:
                        data = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = normalize_voice_phrase(str(result.get("text", "")))
                        if text:
                            command = VoiceCommand(text=text, timestamp=time.time())
                            self.command_queue.put(command)
                            with self.lock:
                                self.last_text = text
                                self.last_heard_at = command.timestamp
        except Exception as exc:
            self._set_status(f"Voice error: {exc}")
        finally:
            self.listening = False
            if self.stop_event.is_set():
                self._set_status("Voice off")

    def _set_status(self, status: str) -> None:
        with self.lock:
            self.status = status


class LivenessChallenge:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.track_id: Optional[int] = None
        self.identity = ""
        self.steps: List[str] = []
        self.step_index = 0
        self.step_started_at = 0.0
        self.hold_started_at = 0.0
        self.passed = False
        self.status = ""

    def update(
        self,
        track: Optional[FaceTrack],
        gesture: GestureFrame,
        hand_ok: bool,
        now: float,
    ) -> bool:
        if not self.enabled:
            self.status = "Liveness disabled"
            return track is not None and track.identity != "Unknown"

        if track is None or track.identity == "Unknown":
            self.reset("Waiting for authorized user")
            return False

        if now - track.last_seen > 0.35:
            self.reset("Liveness reset: controller not visible")
            return False

        if self.track_id != track.track_id or self.identity != track.identity:
            self._start(track, now)

        if self.passed:
            self.status = f"Liveness passed: {track.identity}"
            return True

        if now - self.step_started_at > float(self.config["liveness"]["step_timeout_seconds"]):
            self._start(track, now)
            self.status = "Liveness timeout - new challenge"
            return False

        step = self.steps[self.step_index]
        prompt = LIVENESS_PROMPTS.get(step, step)
        if self._step_satisfied(step, track, gesture, hand_ok):
            if self.hold_started_at == 0.0:
                self.hold_started_at = now
            hold = float(self.config["liveness"]["success_hold_seconds"])
            elapsed = now - self.hold_started_at
            if elapsed >= hold:
                self._advance(track, now)
            else:
                self.status = (
                    f"Liveness {self.step_index + 1}/{len(self.steps)}: "
                    f"hold {prompt} {elapsed:.1f}/{hold:.1f}s"
                )
            return self.passed

        self.hold_started_at = 0.0
        self.status = f"Liveness {self.step_index + 1}/{len(self.steps)}: {prompt}"
        return False

    def reset(self, status: str = "") -> None:
        self.track_id = None
        self.identity = ""
        self.steps = []
        self.step_index = 0
        self.step_started_at = 0.0
        self.hold_started_at = 0.0
        self.passed = False
        self.status = status

    def is_verified_for(self, track: Optional[FaceTrack], now: float) -> bool:
        if not self.enabled:
            return track is not None and track.identity != "Unknown"
        if track is None:
            return False
        return (
            self.passed
            and self.track_id == track.track_id
            and self.identity == track.identity
            and now - track.last_seen <= 0.35
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("liveness", {}).get("enabled", True))

    def _start(self, track: FaceTrack, now: float) -> None:
        self.track_id = track.track_id
        self.identity = track.identity
        self.steps = self._generate_steps()
        self.step_index = 0
        self.step_started_at = now
        self.hold_started_at = 0.0
        self.passed = False
        self.status = f"Liveness required: {LIVENESS_PROMPTS.get(self.steps[0], self.steps[0])}"

    def _generate_steps(self) -> List[str]:
        cfg = self.config["liveness"]
        pool = [
            step
            for step in cfg.get("challenge_pool", [])
            if step in LIVENESS_PROMPTS
        ]
        if not pool:
            pool = list(LIVENESS_PROMPTS.keys())

        count = max(1, int(cfg.get("steps_per_challenge", 2)))
        count = min(count, len(pool))
        head_steps = [step for step in pool if step.startswith("head_")]
        hand_steps = [step for step in pool if step.startswith("hand_")]

        if bool(cfg.get("require_mixed_types", True)) and count >= 2 and head_steps and hand_steps:
            selected = [random.choice(head_steps), random.choice(hand_steps)]
            remaining = [step for step in pool if step not in selected]
            if count > 2 and remaining:
                selected.extend(random.sample(remaining, min(count - 2, len(remaining))))
            random.shuffle(selected)
            return selected[:count]

        return random.sample(pool, count)

    def _advance(self, track: FaceTrack, now: float) -> None:
        self.step_index += 1
        self.hold_started_at = 0.0
        if self.step_index >= len(self.steps):
            self.passed = True
            self.status = f"Liveness passed: {track.identity}"
            return
        prompt = LIVENESS_PROMPTS.get(self.steps[self.step_index], self.steps[self.step_index])
        self.step_started_at = now
        self.status = f"Liveness {self.step_index + 1}/{len(self.steps)}: {prompt}"

    def _step_satisfied(
        self,
        step: str,
        track: FaceTrack,
        gesture: GestureFrame,
        hand_ok: bool,
    ) -> bool:
        if step == "head_left":
            offset = nose_offset_ratio(track)
            return offset is not None and offset < -float(self.config["liveness"]["head_offset_threshold"])
        if step == "head_right":
            offset = nose_offset_ratio(track)
            return offset is not None and offset > float(self.config["liveness"]["head_offset_threshold"])

        expected = LIVENESS_HAND_GESTURES.get(step)
        if expected is None:
            return False
        return (
            hand_ok
            and gesture.name == expected
            and gesture.score >= float(self.config["liveness"]["hand_confidence_threshold"])
        )


class GestureCommandResolver:
    def __init__(
        self,
        config: Dict[str, Any],
        custom_store: CustomGestureStore,
        actions: ActionController,
    ) -> None:
        self.config = config
        self.custom_store = custom_store
        self.actions = actions
        self.candidate = ""
        self.candidate_action = ""
        self.candidate_since = 0.0
        self.last_action_at: Dict[str, float] = {}
        self.last_status = ""

    def process(
        self,
        gesture: GestureFrame,
        authorized: bool,
        hand_ok: bool,
        now: float,
        stats: Optional[PerformanceStats] = None,
    ) -> str:
        if not authorized:
            self._reset_candidate()
            return ""
        if not hand_ok:
            self._reset_candidate()
            return ""

        action = ""
        gesture_name = ""
        custom = self.custom_store.match(
            gesture.landmarks,
            float(self.config["gestures"]["custom_match_threshold"]),
        )
        if custom:
            gesture_name, action, distance = custom
            score = max(0.0, 1.0 - distance)
        else:
            gesture_name = gesture.name
            score = gesture.score
            if gesture_name in ("", "None", "Unknown"):
                self._reset_candidate()
                return ""
            if score < float(self.config["gestures"]["confidence_threshold"]):
                self._reset_candidate()
                return ""
            action = self._configured_action(gesture_name)

        if not action:
            self._reset_candidate()
            return ""

        key = f"{gesture_name}:{action}"
        if key != self.candidate:
            self.candidate = key
            self.candidate_action = action
            self.candidate_since = now
            return f"Gesture: {gesture_name} ({score:.2f})"

        stable_seconds = float(self.config["gestures"]["stable_seconds"])
        cooldown = float(self.config["gestures"]["action_cooldown_seconds"])
        last_at = self.last_action_at.get(action, 0.0)
        if now - self.candidate_since >= stable_seconds and now - last_at >= cooldown:
            self.last_action_at[action] = now
            self.candidate_since = now
            action_start = time.perf_counter()
            self.last_status = self.actions.run_action(action)
            if stats is not None:
                stats.observe("action_ms", (time.perf_counter() - action_start) * 1000.0)
            return self.last_status

        return f"Gesture: {gesture_name} ({score:.2f})"

    def _configured_action(self, gesture_name: str) -> str:
        bindings = self.config["bindings"]
        if gesture_name in bindings.get("global", {}):
            return str(bindings["global"][gesture_name])
        mode_bindings = bindings.get(self.actions.mode, {})
        return str(mode_bindings.get(gesture_name, ""))

    def suppress_action(self, action: str, now: float) -> None:
        if action:
            self.last_action_at[action] = now
        self._reset_candidate()

    def _reset_candidate(self) -> None:
        self.candidate = ""
        self.candidate_action = ""
        self.candidate_since = 0.0


def start_person_capture() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    name_input = simpledialog.askstring("Add Person", "Enter the name of the new person:")
    root.destroy()
    if name_input and name_input.strip():
        return name_input.strip()
    return None


def db_people(db_path: str) -> List[str]:
    if not os.path.exists(db_path):
        return []
    return sorted(
        name
        for name in os.listdir(db_path)
        if os.path.isdir(os.path.join(db_path, name))
    )


def choose_person_to_delete(db_path: str) -> Optional[str]:
    people = db_people(db_path)
    if not people:
        print("No people are registered.")
        return None

    root = tk.Tk()
    root.withdraw()
    person_input = simpledialog.askstring(
        "Delete Person",
        "Enter person name to delete:\n" + ", ".join(people),
    )
    root.destroy()
    if not person_input:
        return None

    normalized = person_input.strip().casefold()
    matches = [person for person in people if person.casefold() == normalized]
    if not matches:
        print(f"Person '{person_input}' was not found.")
        return None
    return matches[0]


def delete_person_folder(db_path: str, person: str) -> bool:
    target = os.path.abspath(os.path.join(db_path, person))
    db_root = os.path.abspath(db_path)
    if not target.startswith(db_root + os.sep):
        print(f"Refusing to delete unsafe path: {target}")
        return False
    if not os.path.isdir(target):
        return False
    shutil.rmtree(target)
    purge_deepface_cache(db_path)
    return True


def start_custom_gesture_capture(actions: ActionController) -> Optional[Tuple[str, str]]:
    valid_actions = ", ".join(actions.valid_actions())
    root = tk.Tk()
    root.withdraw()
    name_input = simpledialog.askstring("Custom Gesture", "Name this gesture:")
    action_input = None
    if name_input and name_input.strip():
        action_input = simpledialog.askstring(
            "Custom Gesture Action",
            f"Assign an action:\n{valid_actions}",
        )
    root.destroy()

    if not name_input or not action_input:
        return None
    name = name_input.strip()
    action = action_input.strip()
    if action not in actions.valid_actions():
        print(f"Unknown action '{action}'. Valid actions: {valid_actions}")
        return None
    return name, action


def purge_deepface_cache(db_path: str) -> None:
    for pkl in glob.glob(os.path.join(db_path, "*.pkl")):
        try:
            os.remove(pkl)
        except OSError:
            pass


def action_label(action: str) -> str:
    return ACTION_LABELS.get(action, action.replace("_", " ").title())


def gesture_label(gesture: str) -> str:
    return GESTURE_LABELS.get(gesture, gesture.replace("_", " "))


def panel_size(
    frame: np.ndarray,
    lines: List[str],
    font_scale: float,
    line_height: int,
    padding: int,
) -> Tuple[int, int]:
    if not lines:
        return 0, 0
    widths = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
        for line in lines
    ]
    panel_w = min(max(widths) + padding * 2, frame.shape[1] - 20)
    panel_h = len(lines) * line_height + padding * 2
    return panel_w, panel_h


def clamp_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    margin: int = 10,
) -> Tuple[int, int, int, int]:
    x = max(margin, min(x, frame.shape[1] - width - margin))
    y = max(margin, min(y, frame.shape[0] - height - margin))
    return x, y, x + width, y + height


def rect_overlap_area(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def expanded_track_rects(
    frame: np.ndarray,
    tracks: List[FaceTrack],
) -> List[Tuple[int, int, int, int]]:
    rects = []
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        rects.append(
            (
                max(0, x1 - 14),
                max(0, y1 - 36),
                min(frame.shape[1], x2 + 14),
                min(frame.shape[0], y2 + 14),
            )
        )
    return rects


def choose_panel_rect(
    frame: np.ndarray,
    panel_w: int,
    panel_h: int,
    reserved: List[Tuple[int, int, int, int]],
    preferred: str,
) -> Tuple[int, int, int, int]:
    margin = 14
    top_y = 66
    candidates = {
        "top_left": (margin, top_y),
        "top_right": (frame.shape[1] - panel_w - margin, top_y),
        "bottom_left": (margin, frame.shape[0] - panel_h - margin),
        "bottom_right": (
            frame.shape[1] - panel_w - margin,
            frame.shape[0] - panel_h - margin,
        ),
    }

    if preferred in candidates:
        x, y = candidates[preferred]
        return clamp_rect(frame, x, y, panel_w, panel_h, margin)

    best_rect = None
    best_score = None
    for name in ("bottom_right", "top_right", "bottom_left", "top_left"):
        x, y = candidates[name]
        rect = clamp_rect(frame, x, y, panel_w, panel_h, margin)
        overlap = sum(rect_overlap_area(rect, item) for item in reserved)
        lower_bonus = 0 if name.startswith("bottom") else 250
        score = overlap + lower_bonus
        if best_score is None or score < best_score:
            best_rect = rect
            best_score = score

    return best_rect if best_rect is not None else clamp_rect(frame, margin, top_y, panel_w, panel_h)


def draw_text_panel(
    frame: np.ndarray,
    lines: List[str],
    x: int,
    y: int,
    alpha: float,
    font_scale: float = 0.46,
    line_height: int = 18,
    padding: int = 8,
) -> Tuple[int, int, int, int]:
    if not lines:
        return 0, 0, 0, 0

    panel_w, panel_h = panel_size(frame, lines, font_scale, line_height, padding)
    x1, y1, x2, y2 = clamp_rect(frame, x, y, panel_w, panel_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (12, 12, 12), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (95, 95, 95), 1)

    text_y = y1 + padding + 13
    for idx, line in enumerate(lines):
        color = (220, 220, 220) if idx else (255, 255, 255)
        cv2.putText(
            frame,
            line,
            (x1 + padding, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
        text_y += line_height
    return x1, y1, x2, y2


def gesture_legend_lines(config: Dict[str, Any], mode: str) -> List[str]:
    bindings = config.get("bindings", {})
    mode_bindings = bindings.get(mode, {})
    lines = [f"{mode.title()} controls"]

    global_bindings = bindings.get("global", {})
    for gesture, action in global_bindings.items():
        lines.append(f"{gesture_label(gesture)} -> {action_label(action)}")

    for gesture, action in mode_bindings.items():
        lines.append(f"{gesture_label(gesture)} -> {action_label(action)}")

    return lines


def draw_legend(
    frame: np.ndarray,
    config: Dict[str, Any],
    mode: str,
    tracks: List[FaceTrack],
) -> Optional[Tuple[int, int, int, int]]:
    if not bool(config.get("ui", {}).get("show_legend", True)):
        return None
    lines = gesture_legend_lines(config, mode)
    alpha = float(config.get("ui", {}).get("overlay_alpha", 0.58))
    font_scale = 0.42
    line_height = 16
    padding = 7
    panel_w, panel_h = panel_size(frame, lines, font_scale, line_height, padding)
    reserved = expanded_track_rects(frame, tracks)
    reserved.append((0, 0, frame.shape[1], 58))
    if bool(config.get("ui", {}).get("show_performance", True)):
        reserved.append((12, frame.shape[0] - 62, 310, frame.shape[0] - 12))
    x1, y1, _, _ = choose_panel_rect(
        frame,
        panel_w,
        panel_h,
        reserved,
        str(config.get("ui", {}).get("legend_position", "auto")),
    )
    return draw_text_panel(
        frame,
        lines,
        x1,
        y1,
        alpha,
        font_scale=font_scale,
        line_height=line_height,
        padding=padding,
    )


def voice_legend_lines(config: Dict[str, Any], mode: str, voice_engine: VoiceCommandEngine) -> List[str]:
    status, last_text, last_action = voice_engine.snapshot()
    lines = [status]
    if last_text:
        lines.append(f"Heard: {last_text}")
    if last_action:
        lines.append(f"Last: {action_label(last_action)}")

    bindings = voice_bindings_for_mode(config, mode)
    max_phrases = int(config.get("voice", {}).get("legend_max_phrases", 9))
    for phrase, action in list(bindings.items())[:max_phrases]:
        lines.append(f"\"{phrase}\" -> {action_label(action)}")
    return lines


def draw_voice_legend(
    frame: np.ndarray,
    config: Dict[str, Any],
    mode: str,
    tracks: List[FaceTrack],
    voice_engine: VoiceCommandEngine,
    extra_reserved: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    if not voice_engine.should_show():
        return None
    lines = voice_legend_lines(config, mode, voice_engine)
    alpha = float(config.get("ui", {}).get("overlay_alpha", 0.58))
    font_scale = 0.40
    line_height = 15
    padding = 7
    panel_w, panel_h = panel_size(frame, lines, font_scale, line_height, padding)
    reserved = expanded_track_rects(frame, tracks)
    reserved.append((0, 0, frame.shape[1], 58))
    if bool(config.get("ui", {}).get("show_performance", True)):
        reserved.append((12, frame.shape[0] - 62, 310, frame.shape[0] - 12))
    if extra_reserved:
        reserved.extend(extra_reserved)
    x1, y1, _, _ = choose_panel_rect(
        frame,
        panel_w,
        panel_h,
        reserved,
        str(config.get("ui", {}).get("voice_legend_position", "auto")),
    )
    return draw_text_panel(
        frame,
        lines,
        x1,
        y1,
        alpha,
        font_scale=font_scale,
        line_height=line_height,
        padding=padding,
    )


def draw_performance(frame: np.ndarray, config: Dict[str, Any], stats: PerformanceStats) -> None:
    if not bool(config.get("ui", {}).get("show_performance", True)):
        return
    lines = [
        f"FPS {stats.fps:.1f} | Detect {stats.detect_ms:.0f} ms",
        (
            f"Track {stats.tracking_ms:.1f} ms | "
            f"Lag {stats.gesture_latency_ms:.0f} | Action {stats.action_ms:.1f}"
        ),
    ]
    alpha = float(config.get("ui", {}).get("overlay_alpha", 0.58))
    font_scale = 0.40
    line_height = 15
    padding = 7
    panel_w, panel_h = panel_size(frame, lines, font_scale, line_height, padding)
    position = str(config.get("ui", {}).get("performance_position", "bottom_left"))
    if position == "bottom_right":
        x = frame.shape[1] - panel_w - 14
    else:
        x = 14
    y = frame.shape[0] - panel_h - 14
    draw_text_panel(
        frame,
        lines,
        x,
        y,
        alpha,
        font_scale=font_scale,
        line_height=line_height,
        padding=padding,
    )


def draw_status(
    frame: np.ndarray,
    tracker: FaceTracker,
    action_controller: ActionController,
    gesture_status: str,
    gesture_engine: MediaPipeGestureEngine,
    liveness: LivenessChallenge,
) -> None:
    now = time.time()
    active = tracker.active_track(now)
    recognized = active is not None and active.identity != "Unknown"
    live_verified = liveness.is_verified_for(active, now)
    if recognized and liveness.enabled and not live_verified:
        color = (0, 220, 255)
        auth_text = f"Recognized: {active.identity} | Liveness required"
    elif recognized:
        color = (0, 220, 0)
        auth_text = tracker.lock_reason
    else:
        color = (0, 0, 255)
        auth_text = tracker.lock_reason
    dry_run = " | DRY RUN" if bool(action_controller.config["external_controls"].get("dry_run", False)) else ""
    status_text = f"{auth_text} | Mode: {action_controller.mode}{dry_run}"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.56, frame, 0.44, 0, frame)
    cv2.putText(
        frame,
        status_text[:78],
        (16, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )
    if not gesture_engine.enabled:
        cv2.putText(
            frame,
            "Gestures disabled: install mediapipe and rerun",
            (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        status_parts = []
        if liveness.enabled and recognized and not live_verified and liveness.status:
            status_parts.append(liveness.status)
        elif liveness.enabled and live_verified and liveness.status:
            status_parts.append(liveness.status)
        if gesture_status and (live_verified or not liveness.enabled or not recognized):
            status_parts.append(gesture_status)
        status = " | ".join(status_parts)
        if not status:
            return
        cv2.putText(
            frame,
            status[:95],
            (16, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_capture_status(
    frame: np.ndarray,
    capture_mode: bool,
    capture_name: str,
    capture_count: int,
    gesture_capture: GestureCaptureState,
    gesture_sample_target: int,
) -> None:
    if capture_mode:
        cv2.putText(
            frame,
            f"CAPTURING {capture_name.upper()}: {capture_count}/5",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255),
            3,
        )
    if gesture_capture.active:
        cv2.putText(
            frame,
            (
                f"CUSTOM {gesture_capture.name}: "
                f"{len(gesture_capture.samples)}/{gesture_sample_target}"
            ),
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )


def draw_faces(frame: np.ndarray, tracks: List[FaceTrack], active_id: Optional[int]) -> None:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        if track.track_id == active_id:
            color = (0, 255, 0)
        elif track.identity != "Unknown":
            color = (0, 180, 0)
        else:
            color = (255, 150, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        emotion = track.emotion if track.emotion else "N/A"
        label = f"{track.identity} [{track.similarity:.2f}] | {emotion.capitalize()}"
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = y1 if y1 > 30 else y1 + 30
        cv2.rectangle(frame, (x1, label_y - 24), (x1 + text_width + 6, label_y), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )


def draw_hand(frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> None:
    if not landmarks:
        return
    h, w = frame.shape[:2]
    for x, y, _ in landmarks:
        cv2.circle(frame, (int(x * w), int(y * h)), 3, (255, 255, 255), -1)


def configure_camera(cap: cv2.VideoCapture, config: Dict[str, Any]) -> None:
    cam_cfg = config["camera"]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg["height"]))
    cap.set(cv2.CAP_PROP_FPS, int(cam_cfg["fps"]))


def open_camera(config: Dict[str, Any]) -> Optional[cv2.VideoCapture]:
    primary = int(config["camera"]["primary_index"])
    fallback = int(config["camera"]["fallback_index"])
    print("Attempting to open the primary webcam...")
    cap = cv2.VideoCapture(primary)
    configure_camera(cap, config)
    if cap.isOpened():
        return cap

    print("Primary webcam failed. Trying fallback webcam...")
    cap.release()
    cap = cv2.VideoCapture(fallback)
    configure_camera(cap, config)
    if cap.isOpened():
        return cap

    cap.release()
    return None


def main() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DB_PATH, exist_ok=True)
    config = load_json_config(CONFIG_PATH, DEFAULT_CONFIG)

    print("Initializing models...")
    if not ensure_file(YUNET_MODEL, YUNET_URL, "YuNet face detector"):
        return
    if not ensure_file(SFACE_MODEL, SFACE_URL, "SFace recognizer"):
        return
    if bool(config["gestures"]["enabled"]):
        ensure_file(GESTURE_MODEL, GESTURE_URL, "MediaPipe gesture recognizer")

    detector = create_detector(config)
    recognizer = create_recognizer()
    if detector is None or recognizer is None:
        return

    face_index = FaceIndex(
        DB_PATH,
        detector,
        recognizer,
        float(config["face"]["sface_cosine_threshold"]),
    )
    face_index.reload()

    tracker = FaceTracker(config, face_index)
    emotion_worker = EmotionWorker(
        bool(config["emotion"]["enabled"]),
        float(config["emotion"]["min_interval_seconds"]),
    )
    custom_store = CustomGestureStore(CUSTOM_GESTURE_PATH)
    gesture_engine = MediaPipeGestureEngine(config)
    motion_gestures = MotionGestureDetector(config)
    action_controller = ActionController(config)
    voice_engine = VoiceCommandEngine(config)
    liveness = LivenessChallenge(config)
    resolver = GestureCommandResolver(config, custom_store, action_controller)
    gesture_capture = GestureCaptureState()
    stats = PerformanceStats()

    cap = open_camera(config)
    if cap is None:
        print("Error: Could not open any webcam stream.")
        emotion_worker.stop()
        gesture_engine.close()
        voice_engine.close()
        return

    print("Stream successfully opened.")
    print("==========================")
    print("KEYS:")
    print("  q -> Quit")
    print("  a -> Add a new authorized person")
    print("  c -> Capture a custom gesture template")
    print("  d -> Delete an authorized person")
    print("  m -> Cycle presentation/video/youtube mode")
    print("  r -> Reload face database")
    print("  t -> Toggle dry-run mode")
    print("  v -> Toggle voice control")
    print("==========================")

    capture_mode = False
    capture_name = ""
    capture_count = 0
    last_cap_time = 0.0
    gesture_status = ""

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            if bool(config["camera"]["mirror"]):
                frame = cv2.flip(frame, 1)

            now = time.time()
            stats.tick(now)
            timestamp_ms = int(now * 1000)
            detect_start = time.perf_counter()
            faces = detect_faces(detector, frame)
            stats.observe("detect_ms", (time.perf_counter() - detect_start) * 1000.0)
            tracking_start = time.perf_counter()
            tracks = tracker.update(frame, faces, now)
            stats.observe("tracking_ms", (time.perf_counter() - tracking_start) * 1000.0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                gesture_status = action_controller.run_action("toggle_mode")
            elif key == ord("r"):
                face_index.reload()
                tracker.reset_identities()
                liveness.reset("Identity database reloaded")
                gesture_status = "Identity database reloaded"
            elif key == ord("t"):
                dry_run = action_controller.toggle_dry_run()
                gesture_status = f"Dry-run {'enabled' if dry_run else 'disabled'}"
            elif key == ord("v"):
                gesture_status = voice_engine.toggle()
            elif key == ord("d") and not capture_mode:
                person = choose_person_to_delete(DB_PATH)
                if person and delete_person_folder(DB_PATH, person):
                    face_index.reload()
                    tracker.reset_identities()
                    liveness.reset(f"Deleted {person}")
                    gesture_status = f"Deleted {person}"
                    print(gesture_status)
            elif key == ord("a") and not capture_mode:
                name = start_person_capture()
                if name:
                    capture_name = name
                    os.makedirs(os.path.join(DB_PATH, capture_name), exist_ok=True)
                    purge_deepface_cache(DB_PATH)
                    capture_mode = True
                    capture_count = 0
                    last_cap_time = now
                    print(f"Starting capture mode for {capture_name}. Tilt your head smoothly.")
            elif key == ord("c") and not gesture_capture.active:
                custom = start_custom_gesture_capture(action_controller)
                if custom:
                    gesture_capture.start(custom[0], custom[1])
                    print(
                        f"Capturing custom gesture '{custom[0]}' for action '{custom[1]}'."
                    )

            if capture_mode and tracks:
                largest = max(tracks, key=lambda track: bbox_area(track.bbox))
                if now - last_cap_time > 0.6:
                    crop = crop_face(frame, largest.bbox)
                    if crop is not None:
                        img_path = os.path.join(
                            DB_PATH,
                            capture_name,
                            f"sample_{capture_count}.jpg",
                        )
                        cv2.imwrite(img_path, crop)
                        capture_count += 1
                        last_cap_time = now
                    if capture_count >= 5:
                        capture_mode = False
                        face_index.reload()
                        tracker.reset_identities()
                        liveness.reset("New identity captured")
                        print(f"Finished capturing 5 profiles for {capture_name}.")

            active = tracker.active_track(now)
            if active is not None:
                active_crop = crop_face(frame, active.bbox)
                emotion_worker.submit(active, active_crop, now)

            for track in tracks:
                track.emotion = emotion_worker.latest(track.track_id)

            recognized = tracker.is_unlocked(now) and not capture_mode
            if recognized:
                gesture_engine.submit(frame, timestamp_ms)

            gesture = gesture_engine.latest()
            if gesture.timestamp_ms:
                stats.observe(
                    "gesture_latency_ms",
                    max(0.0, float(timestamp_ms - gesture.timestamp_ms)),
                )
            hand_ok = tracker.hand_belongs_to_active(gesture.landmarks, frame.shape, now)
            active_for_liveness = tracker.active_track(now) if recognized else None
            liveness_passed = liveness.update(active_for_liveness, gesture, hand_ok, now)
            authorized = recognized and liveness_passed
            motion_gesture = motion_gestures.process(gesture.landmarks, authorized, hand_ok, now)
            command_gesture = motion_gesture if motion_gesture is not None else gesture
            voice_status, voice_action = voice_engine.process_latest(
                action_controller,
                authorized,
                capture_mode or gesture_capture.active,
                now,
                stats=stats,
            )
            if voice_status:
                gesture_status = voice_status
            if voice_action:
                resolver.suppress_action(voice_action, now)
            if gesture_capture.active and authorized and gesture.landmarks:
                sample_target = int(config["gestures"]["custom_capture_samples"])
                interval = float(config["gestures"]["custom_capture_interval_seconds"])
                if now - gesture_capture.last_sample_at >= interval:
                    vector = normalize_landmarks(gesture.landmarks)
                    if vector is not None:
                        gesture_capture.samples.append(vector)
                        gesture_capture.last_sample_at = now
                if len(gesture_capture.samples) >= sample_target:
                    custom_store.add_template(
                        gesture_capture.name,
                        gesture_capture.action,
                        gesture_capture.samples,
                    )
                    gesture_status = (
                        f"Saved custom gesture '{gesture_capture.name}' -> "
                        f"{gesture_capture.action}"
                    )
                    print(gesture_status)
                    gesture_capture.reset()
            elif authorized and not capture_mode and not voice_action:
                status = resolver.process(
                    command_gesture,
                    authorized,
                    hand_ok,
                    now,
                    stats=stats,
                )
                if status:
                    gesture_status = status
            else:
                resolver.process(command_gesture, False, False, now, stats=stats)

            draw_faces(frame, tracks, tracker.active_track_id)
            draw_hand(frame, gesture.landmarks if recognized else [])
            draw_status(frame, tracker, action_controller, gesture_status, gesture_engine, liveness)
            gesture_panel = draw_legend(frame, config, action_controller.mode, tracks)
            draw_voice_legend(
                frame,
                config,
                action_controller.mode,
                tracks,
                voice_engine,
                [gesture_panel] if gesture_panel else None,
            )
            draw_performance(frame, config, stats)
            draw_capture_status(
                frame,
                capture_mode,
                capture_name,
                capture_count,
                gesture_capture,
                int(config["gestures"]["custom_capture_samples"]),
            )

            cv2.imshow("Vision Gesture Media Control", frame)
    finally:
        cap.release()
        emotion_worker.stop()
        gesture_engine.close()
        voice_engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
