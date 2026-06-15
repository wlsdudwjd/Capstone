from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import signal
import sys
import threading
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image

for config_dir in (Path(".ultralytics"), Path(".matplotlib"), Path(".torch")):
    config_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))
os.environ.setdefault("TORCH_HOME", str(Path(".torch").resolve()))

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DET_MODEL_PATH = BASE_DIR / "detection.pt"
DEFAULT_WHOLE_IMAGE_MODEL_PATH = BASE_DIR / "original_best.pt"
DEFAULT_CROP_MODEL_PATH = BASE_DIR / "important_best.pt"
WEB_UI_DIR = BASE_DIR / "web_ui"
BOUNDARY = "frame"
DEFAULT_VIOLATION_LOG_DIR = BASE_DIR / "violation_logs"
VIOLATION_LOG_FILENAME = "violation_events.csv"
DEFAULT_STICKER_ONLY_MARGIN_RATIO = 0.4
DEFAULT_CROP_DISPLAY_SMOOTHING = 0.35
VIOLATION_LOG_FIELDS = [
    "timestamp",
    "date",
    "time",
    "overall_final",
    "overall_reason",
    "front_final",
    "front_whole_pred",
    "front_whole_conf",
    "front_crop_pred",
    "front_crop_conf",
    "front_crop_reason",
    "front_fused_reason",
    "front_phones",
    "front_lenses",
    "front_stickers",
    "front_void_pred",
    "front_void_conf",
    "front_void_reason",
    "back_final",
    "back_whole_pred",
    "back_whole_conf",
    "back_crop_pred",
    "back_crop_conf",
    "back_crop_reason",
    "back_fused_reason",
    "back_phones",
    "back_lenses",
    "back_stickers",
    "back_void_pred",
    "back_void_conf",
    "back_void_reason",
    "snapshot_path",
    "front_crop_path",
    "back_crop_path",
]


class LetterboxPad:
    def __init__(
        self,
        size: int | tuple[int, int],
        fill: tuple[int, int, int] = (114, 114, 114),
    ):
        if isinstance(size, int):
            self.target_h = size
            self.target_w = size
        else:
            self.target_h, self.target_w = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        src_w, src_h = image.size
        scale = min(self.target_w / src_w, self.target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.target_w, self.target_h), self.fill)
        offset_x = (self.target_w - new_w) // 2
        offset_y = (self.target_h - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


def register_legacy_checkpoint_shims() -> None:
    module_name = "train_sticker_yolo_cls"
    if module_name not in sys.modules:
        shim = types.ModuleType(module_name)
        shim.LetterboxPad = LetterboxPad
        sys.modules[module_name] = shim

    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "LetterboxPad"):
        setattr(main_module, "LetterboxPad", LetterboxPad)


def normalize_text(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def resolve_name(names, index: int) -> str:
    if isinstance(names, dict):
        return str(names[index])
    return str(names[index])


def normalize_normal_violation_label(label: str) -> str:
    normalized = normalize_text(label)
    if normalized == "normal":
        return "normal"
    if normalized == "violation":
        return "violation"
    return "unknown"


def crop_around_boxes(
    image: np.ndarray,
    boxes: list[list[float]],
    margin_ratio: float,
) -> tuple[np.ndarray | None, list[int] | None]:
    if not boxes:
        return None, None

    image_height, image_width = image.shape[:2]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)

    width = x2 - x1
    height = y2 - y1
    margin_x = width * margin_ratio
    margin_y = height * margin_ratio

    crop_x1 = max(0, int(round(x1 - margin_x)))
    crop_y1 = max(0, int(round(y1 - margin_y)))
    crop_x2 = min(image_width, int(round(x2 + margin_x)))
    crop_y2 = min(image_height, int(round(y2 + margin_y)))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None, None

    return image[crop_y1:crop_y2, crop_x1:crop_x2], [crop_x1, crop_y1, crop_x2, crop_y2]


def build_image_level_lens_sticker_crop(
    image: np.ndarray,
    lens_items: list[dict[str, object]],
    sticker_items: list[dict[str, object]],
    margin_ratio: float,
    sticker_only_margin_ratio: float,
) -> tuple[np.ndarray | None, list[int] | None, str]:
    if sticker_items:
        if lens_items:
            crop_boxes = [lens_item["box"] for lens_item in lens_items] + [
                sticker_item["box"] for sticker_item in sticker_items
            ]
            crop, crop_box = crop_around_boxes(image=image, boxes=crop_boxes, margin_ratio=margin_ratio)
            return crop, crop_box, "all_lens_and_all_stickers"

        sticker_boxes = [sticker_item["box"] for sticker_item in sticker_items]
        crop, crop_box = crop_around_boxes(
            image=image,
            boxes=sticker_boxes,
            margin_ratio=sticker_only_margin_ratio,
        )
        return crop, crop_box, "all_stickers_only_context"

    if not lens_items:
        return None, None, "no_lens"

    lens_boxes = [lens_item["box"] for lens_item in lens_items]
    crop, crop_box = crop_around_boxes(image=image, boxes=lens_boxes, margin_ratio=margin_ratio)
    return crop, crop_box, "all_lenses_only"


def is_detection_class(label: str, class_name: str) -> bool:
    return normalize_text(label) == class_name


def extract_boxes_from_detection(result, names) -> dict[str, list[dict[str, object]]]:
    grouped = {"phone": [], "lens": [], "sticker": []}

    if result.boxes is None:
        return grouped

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = resolve_name(names, class_id)
        confidence = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        item = {"box": xyxy, "conf": confidence, "class_name": class_name}

        if is_detection_class(class_name, "phone"):
            grouped["phone"].append(item)
        elif is_detection_class(class_name, "lens"):
            grouped["lens"].append(item)
        elif is_detection_class(class_name, "sticker"):
            grouped["sticker"].append(item)

    for items in grouped.values():
        items.sort(key=lambda value: float(value["conf"]), reverse=True)

    return grouped


def predict_top1_label(
    model: YOLO,
    crop: np.ndarray,
    imgsz: int,
    device: str,
    normalizer,
    model_lock: threading.Lock | None = None,
) -> tuple[str | None, str, float]:
    if model_lock is None:
        result = model.predict(source=crop, imgsz=imgsz, device=device, verbose=False)[0]
    else:
        with model_lock:
            result = model.predict(source=crop, imgsz=imgsz, device=device, verbose=False)[0]
    probs = result.probs
    if probs is None:
        return None, "unknown", 0.0

    top1_index = int(probs.top1)
    confidence = float(probs.top1conf)
    raw_label = resolve_name(model.names, top1_index)
    return raw_label, normalizer(raw_label), confidence


def choose_fused_label(
    whole_pred: str,
    whole_conf: float,
    crop_pred: str,
    crop_reason: str,
    fusion: str,
    whole_conf_threshold: float,
    crop_conf: float,
    crop_conf_threshold: float,
    allow_review: bool,
) -> tuple[str, str]:
    whole_valid = whole_pred in {"normal", "violation"} and whole_conf >= whole_conf_threshold
    crop_valid = crop_pred in {"normal", "violation"} and crop_conf >= crop_conf_threshold

    def finalize(label: str, reason: str) -> tuple[str, str]:
        return label, reason

    def resolve_no_review(default_reason: str) -> tuple[str, str]:
        if crop_pred in {"normal", "violation"}:
            return finalize(crop_pred, f"forced_{default_reason}_crop_raw")
        if whole_pred in {"normal", "violation"}:
            return finalize(whole_pred, f"forced_{default_reason}_whole_raw")
        return finalize("violation", f"forced_{default_reason}_default_violation")

    if fusion == "whole_image_priority":
        if whole_valid:
            return finalize(whole_pred, f"whole_image_priority_{whole_pred}")
        if crop_valid:
            return finalize(crop_pred, f"crop_fallback_{crop_reason}")
        return ("review", "both_invalid") if allow_review else resolve_no_review("both_invalid")

    if fusion == "crop_priority":
        if crop_valid:
            return finalize(crop_pred, f"crop_priority_{crop_reason}")
        if whole_valid:
            return finalize(whole_pred, f"whole_image_fallback_{whole_pred}")
        return ("review", "both_invalid") if allow_review else resolve_no_review("both_invalid")

    if fusion == "violation_priority":
        if whole_valid and whole_pred == "violation":
            return finalize("violation", "whole_image_violation_priority")
        if crop_valid and crop_pred == "violation":
            return finalize("violation", f"crop_violation_priority_{crop_reason}")
        if whole_valid and crop_valid and whole_pred == crop_pred:
            return finalize(whole_pred, f"agreement_{whole_pred}")
        if whole_valid and not crop_valid:
            return finalize(whole_pred, f"whole_image_only_{whole_pred}")
        if crop_valid and not whole_valid:
            return finalize(crop_pred, f"crop_only_{crop_reason}")
        return ("review", "disagreement_or_invalid") if allow_review else resolve_no_review("disagreement_or_invalid")

    if whole_valid and crop_valid and whole_pred == crop_pred:
        return finalize(whole_pred, f"agreement_{whole_pred}")
    if whole_valid and not crop_valid:
        if allow_review:
            return "review", f"crop_review_whole_{whole_pred}"
        return finalize(whole_pred, f"forced_crop_review_whole_{whole_pred}")
    if crop_valid and not whole_valid:
        if allow_review:
            return "review", f"whole_invalid_crop_{crop_pred}"
        return finalize(crop_pred, f"forced_whole_invalid_crop_{crop_pred}")
    return ("review", "disagreement_or_invalid") if allow_review else resolve_no_review("disagreement_or_invalid")


def draw_detection_boxes_from_status(image: np.ndarray, status: dict[str, object]) -> np.ndarray:
    boxed_image = image.copy()
    detections = status.get("detections", {"phone": [], "lens": [], "sticker": []})
    colors = {"phone": (255, 0, 0), "lens": (0, 0, 255), "sticker": (0, 255, 0)}

    def draw_box_label(label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        font_scale = 0.42
        thickness = 1
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size
        label_x = max(0, x)
        label_y = max(text_height + baseline + 4, y)
        cv2.rectangle(
            boxed_image,
            (label_x, label_y - text_height - baseline - 4),
            (label_x + text_width + 6, label_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            boxed_image,
            label,
            (label_x + 3, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    if isinstance(detections, dict):
        for class_name, color in colors.items():
            items = detections.get(class_name, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict) or "box" not in item:
                    continue
                x1, y1, x2, y2 = map(int, item["box"])
                cv2.rectangle(boxed_image, (x1, y1), (x2, y2), color, 2)
                draw_box_label(class_name, x1, y1, color)

    crop_box = status.get("crop_box")
    if crop_box is not None:
        x1, y1, x2, y2 = map(int, crop_box)
        cv2.rectangle(boxed_image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return boxed_image


def resize_and_cover(frame, target_width: int, target_height: int):
    if frame is None:
        return np.full((target_height, target_width, 3), 30, dtype=np.uint8)

    source_height, source_width = frame.shape[:2]
    if source_height <= 0 or source_width <= 0:
        return np.full((target_height, target_width, 3), 30, dtype=np.uint8)

    scale = max(target_width / source_width, target_height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

    offset_x = max(0, (resized_width - target_width) // 2)
    offset_y = max(0, (resized_height - target_height) // 2)
    return resized[offset_y : offset_y + target_height, offset_x : offset_x + target_width].copy()


def draw_panel_title(frame, title: str):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (15, 15, 15), -1)
    cv2.putText(
        frame,
        title,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame


def validate_paths(args: argparse.Namespace) -> None:
    for path in (args.whole_image_model, args.det_model, args.crop_normal_violation_model):
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")


def initialize_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")
    return capture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the webcam normal/violation pipeline in a local web dashboard."
    )
    parser.add_argument("--whole-image-model", type=Path, default=DEFAULT_WHOLE_IMAGE_MODEL_PATH)
    parser.add_argument("--det-model", type=Path, default=DEFAULT_DET_MODEL_PATH)
    parser.add_argument("--crop-normal-violation-model", type=Path, default=DEFAULT_CROP_MODEL_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Open one camera. If omitted, the dashboard opens camera indexes 0 and 1.",
    )
    parser.add_argument(
        "--camera-indexes",
        type=int,
        nargs="+",
        default=None,
        help="Open multiple camera indexes, for example: --camera-indexes 0 1",
    )
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--whole-imgsz", type=int, default=224)
    parser.add_argument("--det-conf", type=float, default=0.15)
    parser.add_argument("--det-imgsz", type=int, default=960)
    parser.add_argument("--crop-imgsz", type=int, default=224)
    parser.add_argument("--whole-conf-threshold", type=float, default=0.0)
    parser.add_argument("--crop-conf-threshold", type=float, default=0.0)
    parser.add_argument(
        "--fusion",
        choices=("agree_or_review", "violation_priority", "whole_image_priority", "crop_priority"),
        default="crop_priority",
    )
    parser.add_argument("--allow-review", action="store_true")
    parser.add_argument("--lens-sticker-margin-ratio", type=float, default=0.30)
    parser.add_argument(
        "--sticker-only-margin-ratio",
        type=float,
        default=DEFAULT_STICKER_ONLY_MARGIN_RATIO,
        help="Crop margin used when stickers are detected but lenses are not.",
    )
    parser.add_argument(
        "--crop-display-smoothing",
        type=float,
        default=DEFAULT_CROP_DISPLAY_SMOOTHING,
        help="EMA smoothing factor for displayed crop normal/violation confidence. 1 disables smoothing.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Skip this many frames between inferences. 0 means infer every frame.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument(
        "--stream-fps",
        type=float,
        default=15.0,
        help="Maximum FPS for each MJPEG stream. Inference still runs according to --frame-skip.",
    )
    parser.add_argument("--violation-log-dir", type=Path, default=DEFAULT_VIOLATION_LOG_DIR)
    parser.add_argument("--violation-log-cooldown-seconds", type=float, default=5.0)
    parser.add_argument(
        "--no-auto-log-violations",
        action="store_false",
        dest="auto_log_violations",
        help="Disable automatic violation CSV/image logging.",
    )
    parser.set_defaults(auto_log_violations=True)
    return parser.parse_args()


def camera_indexes_from_args(args: argparse.Namespace) -> list[int]:
    if args.camera_indexes is not None:
        raw_indexes = args.camera_indexes
    elif args.camera_index is not None:
        raw_indexes = [args.camera_index]
    else:
        raw_indexes = [0, 1]

    indexes: list[int] = []
    for camera_index in raw_indexes:
        if camera_index not in indexes:
            indexes.append(camera_index)
    return indexes


def absolute_path(path: Path) -> Path:
    return path.expanduser().resolve()


def build_camera_error(camera_index: int) -> str:
    return f"Could not open webcam index {camera_index}"


def run_inference_on_frame(
    *,
    args: argparse.Namespace,
    frame: np.ndarray,
    whole_model: YOLO,
    det_model: YOLO,
    crop_model: YOLO,
    model_lock: threading.Lock | None = None,
) -> tuple[np.ndarray | None, dict[str, object]]:
    start_time = time.perf_counter()

    if model_lock is None:
        det_result = det_model.predict(
            source=frame,
            conf=args.det_conf,
            imgsz=args.det_imgsz,
            device=args.device,
            verbose=False,
        )[0]
    else:
        with model_lock:
            det_result = det_model.predict(
                source=frame,
                conf=args.det_conf,
                imgsz=args.det_imgsz,
                device=args.device,
                verbose=False,
            )[0]
    detections = extract_boxes_from_detection(det_result, det_model.names)

    whole_raw_pred, whole_pred, whole_conf = predict_top1_label(
        model=whole_model,
        crop=frame,
        imgsz=args.whole_imgsz,
        device=args.device,
        normalizer=normalize_normal_violation_label,
        model_lock=model_lock,
    )

    def build_status(
        *,
        whole_raw_pred: str | None = None,
        whole_pred: str = "unknown",
        whole_conf: float = 0.0,
        crop_raw_pred: str | None = None,
        crop_pred: str = "unknown",
        crop_conf: float = 0.0,
        crop_reason: str,
        fused_final: str,
        fused_reason: str,
        crop_box: list[int] | None = None,
    ) -> dict[str, object]:
        elapsed = time.perf_counter() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        return {
            "whole_raw_pred": whole_raw_pred,
            "whole_pred": whole_pred,
            "whole_conf": whole_conf,
            "crop_raw_pred": crop_raw_pred,
            "crop_pred": crop_pred,
            "crop_conf": crop_conf,
            "crop_reason": crop_reason,
            "fused_final": fused_final,
            "fused_reason": fused_reason,
            "phones": len(detections["phone"]),
            "lenses": len(detections["lens"]),
            "stickers": len(detections["sticker"]),
            "fps": fps,
            "detections": detections,
            "crop_box": crop_box,
        }

    if len(detections["phone"]) == 0:
        crop_reason = "no_phone_detected"
        fused_final = "no_detection"
        fused_reason = "no_phone_detected"
        return None, build_status(
            whole_raw_pred=whole_raw_pred,
            whole_pred=whole_pred,
            whole_conf=whole_conf,
            crop_reason=crop_reason,
            fused_final=fused_final,
            fused_reason=fused_reason,
        )

    if len(detections["phone"]) > 0 and len(detections["sticker"]) == 0:
        crop_reason = "no_sticker_detected"
        fused_final = "violation"
        fused_reason = "no_sticker_detected"
        return None, build_status(
            whole_raw_pred=whole_raw_pred,
            whole_pred=whole_pred,
            whole_conf=whole_conf,
            crop_reason=crop_reason,
            fused_final=fused_final,
            fused_reason=fused_reason,
        )

    crop, crop_box, crop_reason = build_image_level_lens_sticker_crop(
        image=frame,
        lens_items=detections["lens"],
        sticker_items=detections["sticker"],
        margin_ratio=args.lens_sticker_margin_ratio,
        sticker_only_margin_ratio=args.sticker_only_margin_ratio,
    )

    crop_raw_pred = None
    crop_pred = "unknown"
    crop_conf = 0.0
    if crop is not None and crop.size > 0:
        crop_raw_pred, crop_pred, crop_conf = predict_top1_label(
            model=crop_model,
            crop=crop,
            imgsz=args.crop_imgsz,
            device=args.device,
            normalizer=normalize_normal_violation_label,
            model_lock=model_lock,
        )
    else:
        crop_reason = "no_lens_detected"

    fused_final, fused_reason = choose_fused_label(
        whole_pred=whole_pred,
        whole_conf=whole_conf,
        crop_pred=crop_pred,
        crop_reason=crop_reason,
        fusion=args.fusion,
        whole_conf_threshold=args.whole_conf_threshold,
        crop_conf=crop_conf,
        crop_conf_threshold=args.crop_conf_threshold,
        allow_review=args.allow_review,
    )

    status = build_status(
        whole_raw_pred=whole_raw_pred,
        whole_pred=whole_pred,
        whole_conf=whole_conf,
        crop_raw_pred=crop_raw_pred,
        crop_pred=crop_pred,
        crop_conf=crop_conf,
        crop_reason=crop_reason,
        fused_final=fused_final,
        fused_reason=fused_reason,
        crop_box=crop_box,
    )
    return crop, status


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    return initialize_camera(camera_index, width, height)


def validate_web_paths(args: argparse.Namespace) -> None:
    validate_paths(args)


def choose_overall_label(
    front_status: dict[str, object],
    back_status: dict[str, object],
    allow_review: bool,
) -> tuple[str, str]:
    front_phones = status_int(front_status, "phones", 0)
    back_phones = status_int(back_status, "phones", 0)
    if front_phones <= 0 or back_phones <= 0:
        return "no_detection", "front_back_phone_required"

    finals = [
        str(status_value(front_status, "fused_final", "unknown")).lower(),
        str(status_value(back_status, "fused_final", "unknown")).lower(),
    ]
    if "violation" in finals:
        return "violation", "any_camera_violation"
    if all(value == "normal" for value in finals):
        return "normal", "all_cameras_normal"
    if allow_review and "review" in finals:
        return "review", "some_camera_review"
    return "no_detection", "some_camera_no_detection"


def build_camera_view(
    front_frame: np.ndarray,
    back_frame: np.ndarray,
    front_status: dict[str, object],
    back_status: dict[str, object],
    front_name: str,
    back_name: str,
) -> np.ndarray:
    panel_width = 640
    panel_height = 360
    front_panel = draw_panel_title(
        resize_and_cover(front_frame, panel_width, panel_height),
        f"{front_name}: {status_value(front_status, 'fused_final', 'unknown')}",
    )
    back_panel = draw_panel_title(
        resize_and_cover(back_frame, panel_width, panel_height),
        f"{back_name}: {status_value(back_status, 'fused_final', 'unknown')}",
    )
    return cv2.hconcat([front_panel, back_panel])


def build_crop_view(
    front_crop: np.ndarray | None,
    back_crop: np.ndarray | None,
    front_name: str,
    back_name: str,
    output_width: int,
) -> np.ndarray:
    panel_width = max(1, output_width // 2)
    panel_height = 220
    front_panel = draw_panel_title(
        resize_and_cover(front_crop, panel_width, panel_height),
        f"{front_name} Crop",
    )
    back_panel = draw_panel_title(
        resize_and_cover(back_crop, panel_width, panel_height),
        f"{back_name} Crop",
    )
    return cv2.hconcat([front_panel, back_panel])


def build_combined_view(
    camera_view: np.ndarray,
    crop_view: np.ndarray,
    overall_final: str,
    overall_reason: str,
) -> np.ndarray:
    combined = cv2.vconcat([camera_view, crop_view])
    color = (0, 200, 0)
    if overall_final == "violation":
        color = (0, 0, 255)
    elif overall_final in {"no_detection", "review"}:
        color = (0, 165, 255)
    cv2.rectangle(combined, (0, 0), (combined.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(
        combined,
        f"GLOBAL: {overall_final} / {overall_reason}",
        (14, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
    )
    return combined


def make_daily_violation_paths(root_dir: Path, captured_at: datetime) -> dict[str, Path]:
    day_dir = root_dir / captured_at.strftime("%Y-%m-%d")
    return {
        "day_dir": day_dir,
        "csv": day_dir / VIOLATION_LOG_FILENAME,
        "snapshots": day_dir / "snapshots",
        "crops": day_dir / "crops",
    }


def write_violation_image(path: Path, image: np.ndarray | None) -> str:
    if image is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return str(path)


def status_value(status: dict[str, object] | None, key: str, default: object = "") -> object:
    if status is None:
        return default
    return status.get(key, default)


def status_int(status: dict[str, object] | None, key: str, default: int = 0) -> int:
    try:
        return int(status_value(status, key, default))
    except (TypeError, ValueError):
        return default


def format_conf(status: dict[str, object] | None, key: str) -> str:
    return f"{float(status_value(status, key, 0.0)):.4f}"


def has_valid_object_detection(status: dict[str, object]) -> bool:
    phones = status_int(status, "phones", 0)
    lenses = status_int(status, "lenses", 0)
    stickers = status_int(status, "stickers", 0)
    crop_pred = str(status_value(status, "crop_pred", "unknown")).lower()
    crop_reason = str(status_value(status, "crop_reason", "")).lower()
    if phones > 0 and crop_reason == "no_sticker_detected":
        return True
    return (
        phones > 0
        and (lenses > 0 or stickers > 0)
        and crop_pred not in {"unknown", "waiting"}
        and crop_reason != "no_lens_detected"
    )


def has_loggable_violation(*statuses: dict[str, object]) -> bool:
    return any(
        str(status_value(status, "fused_final", "")).lower() == "violation"
        and has_valid_object_detection(status)
        for status in statuses
    )


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    quality = max(1, min(100, int(quality)))
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def make_placeholder_frame(title: str, message: str, width: int = 960, height: int = 540) -> np.ndarray:
    frame = np.full((height, width, 3), (250, 252, 255), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (230, 238, 247), 2)
    cv2.rectangle(frame, (28, 28), (width - 28, height - 28), (255, 255, 255), -1)
    cv2.rectangle(frame, (28, 28), (width - 28, height - 28), (220, 232, 245), 1)
    cv2.putText(frame, title, (56, 92), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (168, 88, 18), 2)

    y = 142
    for line in wrap_text(message, max_chars=54):
        cv2.putText(frame, line, (56, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (82, 95, 112), 2)
        y += 34
    return frame


def make_blank_frame(width: int = 960, height: int = 540) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_blank_crop_frame(width: int = 960, height: int = 420) -> np.ndarray:
    return make_blank_frame(width=width, height=height)


def wrap_text(value: str, max_chars: int) -> list[str]:
    words = value.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines or [""]


def make_initial_status(camera_index: int) -> dict[str, object]:
    return {
        "camera_index": camera_index,
        "active": False,
        "error": None,
        "whole_pred": "waiting",
        "whole_conf": 0.0,
        "crop_pred": "waiting",
        "crop_conf": 0.0,
        "crop_reason": "waiting_for_camera",
        "display_crop_pred": "waiting",
        "display_crop_conf": 0.0,
        "fused_final": "waiting",
        "fused_reason": "waiting_for_camera",
        "phones": 0,
        "lenses": 0,
        "stickers": 0,
        "fps": 0.0,
        "updated_at": None,
    }


class CropDisplaySmoother:
    def __init__(self, smoothing: float):
        self.smoothing = max(0.0, min(1.0, float(smoothing)))
        self.score: float | None = None

    def update(self, pred: str, conf: float) -> tuple[str, float]:
        pred = str(pred or "unknown").lower()
        try:
            conf = max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            conf = 0.0

        if pred not in {"normal", "violation"}:
            self.score = None
            return pred, conf

        raw_score = conf if pred == "violation" else 1.0 - conf
        if self.score is None:
            self.score = raw_score
        else:
            alpha = self.smoothing
            self.score = (alpha * raw_score) + ((1.0 - alpha) * self.score)

        if self.score >= 0.5:
            return "violation", self.score
        return "normal", 1.0 - self.score


class FrameStore:
    def __init__(self, camera_index: int, jpeg_quality: int, crop_display_smoothing: float):
        self.camera_index = camera_index
        self.jpeg_quality = jpeg_quality
        self.crop_display_smoother = CropDisplaySmoother(crop_display_smoothing)
        self.condition = threading.Condition()
        self.full_seq = 0
        self.crop_seq = 0
        self.blank_crop_jpeg = encode_jpeg(make_blank_crop_frame(), jpeg_quality)
        self.full_jpeg = encode_jpeg(
            make_placeholder_frame(f"CAM {camera_index} Full", "Waiting for camera stream."),
            jpeg_quality,
        )
        self.crop_jpeg = self.blank_crop_jpeg
        self.full_frame: np.ndarray | None = None
        self.crop_frame: np.ndarray | None = None
        self.status = make_initial_status(camera_index)

    def update(
        self,
        full_frame: np.ndarray,
        crop_frame: np.ndarray | None,
        status: dict[str, object],
        update_crop: bool = True,
    ) -> None:
        full_jpeg = encode_jpeg(full_frame, self.jpeg_quality)
        if update_crop:
            next_crop_frame = None if crop_frame is None or crop_frame.size == 0 else crop_frame
            crop_jpeg = (
                self.blank_crop_jpeg
                if next_crop_frame is None
                else encode_jpeg(next_crop_frame, self.jpeg_quality)
            )

        next_status = make_initial_status(self.camera_index)
        next_status.update(status)
        next_status["camera_index"] = self.camera_index
        next_status["active"] = True
        next_status["error"] = None
        next_status["updated_at"] = time.time()

        with self.condition:
            if update_crop:
                display_crop_pred, display_crop_conf = self.crop_display_smoother.update(
                    str(next_status.get("crop_pred", "unknown")),
                    float(next_status.get("crop_conf", 0.0)),
                )
            else:
                display_crop_pred = str(
                    status_value(
                        self.status,
                        "display_crop_pred",
                        status_value(self.status, "crop_pred", "waiting"),
                    )
                )
                try:
                    display_crop_conf = float(
                        status_value(
                            self.status,
                            "display_crop_conf",
                            status_value(self.status, "crop_conf", 0.0),
                        )
                    )
                except (TypeError, ValueError):
                    display_crop_conf = 0.0
            next_status["display_crop_pred"] = display_crop_pred
            next_status["display_crop_conf"] = display_crop_conf
            self.full_jpeg = full_jpeg
            self.full_frame = full_frame
            if update_crop:
                self.crop_jpeg = crop_jpeg
                self.crop_frame = next_crop_frame
                self.crop_seq += 1
            self.status = next_status
            self.full_seq += 1
            self.condition.notify_all()

    def set_error(self, message: str) -> None:
        full_frame = make_blank_frame()
        crop_frame = make_blank_crop_frame()
        full_jpeg = encode_jpeg(full_frame, self.jpeg_quality)
        crop_jpeg = encode_jpeg(crop_frame, self.jpeg_quality)
        next_status = make_initial_status(self.camera_index)
        next_status.update(
            {
                "active": False,
                "error": message,
                "fused_final": "error",
                "fused_reason": "camera_error",
                "updated_at": time.time(),
            }
        )

        with self.condition:
            self.crop_display_smoother.score = None
            next_status["display_crop_pred"] = "error"
            next_status["display_crop_conf"] = 0.0
            self.full_jpeg = full_jpeg
            self.crop_jpeg = crop_jpeg
            self.full_frame = full_frame
            self.crop_frame = crop_frame
            self.status = next_status
            self.full_seq += 1
            self.crop_seq += 1
            self.condition.notify_all()

    def wait_for_jpeg(self, kind: str, last_seq: int, timeout: float = 2.0) -> tuple[bytes, int]:
        with self.condition:
            current_seq = self.full_seq if kind == "full" else self.crop_seq
            if current_seq == last_seq:
                self.condition.wait(timeout=timeout)

            if kind == "full":
                return self.full_jpeg, self.full_seq
            return self.crop_jpeg, self.crop_seq

    def snapshot_status(self) -> dict[str, object]:
        with self.condition:
            return dict(self.status)

    def snapshot_for_logging(self) -> tuple[dict[str, object], np.ndarray | None, np.ndarray | None]:
        with self.condition:
            full_frame = None if self.full_frame is None else self.full_frame.copy()
            crop_frame = None if self.crop_frame is None else self.crop_frame.copy()
            return dict(self.status), full_frame, crop_frame


class CameraWorker(threading.Thread):
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        camera_index: int,
        frame_store: FrameStore,
        whole_model: YOLO,
        det_model: YOLO,
        crop_model: YOLO,
        infer_lock: threading.Lock,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.args = args
        self.camera_index = camera_index
        self.frame_store = frame_store
        self.whole_model = whole_model
        self.det_model = det_model
        self.crop_model = crop_model
        self.infer_lock = infer_lock
        self.stop_event = stop_event
        self.publish_interval = 0.0 if args.stream_fps <= 0 else 1.0 / args.stream_fps

    def run(self) -> None:
        try:
            cap = open_camera(self.camera_index, self.args.camera_width, self.args.camera_height)
        except Exception as exc:  # noqa: BLE001
            self.frame_store.set_error(f"{build_camera_error(self.camera_index)}: {exc}")
            return

        frame_index = 0
        inference_future: Future | None = None
        executor = ThreadPoolExecutor(max_workers=1)
        last_crop_frame = None
        last_status = make_initial_status(self.camera_index)
        last_publish_time = 0.0

        def infer_frame_for_camera(frame_to_infer):
            return run_inference_on_frame(
                args=self.args,
                frame=frame_to_infer,
                whole_model=self.whole_model,
                det_model=self.det_model,
                crop_model=self.crop_model,
                model_lock=self.infer_lock,
            )

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.frame_store.set_error(f"Camera {self.camera_index}: frame read failed.")
                    break

                inference_updated = False
                if inference_future is not None and inference_future.done():
                    try:
                        crop_frame, status = inference_future.result()
                        last_crop_frame = crop_frame.copy() if crop_frame is not None and crop_frame.size > 0 else None
                        last_status = status
                        inference_updated = True
                    except Exception as exc:  # noqa: BLE001
                        self.frame_store.set_error(f"Camera {self.camera_index}: inference failed: {exc}")
                        break
                    finally:
                        inference_future = None

                should_infer = (
                    inference_future is None
                    and (self.args.frame_skip == 0 or frame_index % (self.args.frame_skip + 1) == 0)
                )
                if should_infer:
                    inference_future = executor.submit(infer_frame_for_camera, frame.copy())

                now = time.monotonic()
                should_publish = (
                    inference_updated
                    or last_publish_time == 0.0
                    or self.publish_interval == 0.0
                    or now - last_publish_time >= self.publish_interval
                )
                if should_publish:
                    live_frame = draw_detection_boxes_from_status(frame, last_status)
                    self.frame_store.update(
                        live_frame,
                        last_crop_frame,
                        last_status,
                        update_crop=inference_updated,
                    )
                    last_publish_time = now

                frame_index += 1
        finally:
            if inference_future is not None:
                inference_future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            cap.release()


class Dashboard:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.camera_indexes = camera_indexes_from_args(args)
        self.stop_event = threading.Event()
        self.infer_lock = threading.Lock()
        self.frame_stores = {
            camera_index: FrameStore(
                camera_index,
                args.jpeg_quality,
                args.crop_display_smoothing,
            )
            for camera_index in self.camera_indexes
        }
        self.workers: list[CameraWorker] = []
        self.log_thread: threading.Thread | None = None
        self.log_lock = threading.Lock()
        self.last_violation_log_time = 0.0
        self.last_violation_log_message = ""
        self.violation_event_logged = False

        register_legacy_checkpoint_shims()
        main_module = sys.modules.get("__main__")
        if main_module is not None and not hasattr(main_module, "LetterboxPad"):
            setattr(main_module, "LetterboxPad", LetterboxPad)

        self.whole_model = YOLO(str(args.whole_image_model), task="classify")
        self.det_model = YOLO(str(args.det_model))
        self.crop_model = YOLO(str(args.crop_normal_violation_model), task="classify")

    def start(self) -> None:
        for camera_index in self.camera_indexes:
            worker = CameraWorker(
                args=self.args,
                camera_index=camera_index,
                frame_store=self.frame_stores[camera_index],
                whole_model=self.whole_model,
                det_model=self.det_model,
                crop_model=self.crop_model,
                infer_lock=self.infer_lock,
                stop_event=self.stop_event,
            )
            worker.start()
            self.workers.append(worker)
        if self.args.auto_log_violations:
            self.log_thread = threading.Thread(target=self.log_violations_loop, daemon=True)
            self.log_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        for store in self.frame_stores.values():
            with store.condition:
                store.condition.notify_all()
        for worker in self.workers:
            worker.join(timeout=2.0)
        if self.log_thread is not None:
            self.log_thread.join(timeout=2.0)

    def log_violations_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.maybe_log_violation()
            except Exception as exc:  # noqa: BLE001
                self.last_violation_log_message = f"violation log failed: {exc}"
                print(self.last_violation_log_message, flush=True)
            self.stop_event.wait(timeout=0.5)

    def maybe_log_violation(self) -> None:
        if len(self.camera_indexes) < 2:
            return

        front_index, back_index = self.camera_indexes[:2]
        front_status = self.frame_stores[front_index].snapshot_status()
        back_status = self.frame_stores[back_index].snapshot_status()
        if not front_status.get("active") or not back_status.get("active"):
            self.last_violation_log_message = "waiting for both cameras to become active before saving violation"
            return

        overall_final, overall_reason = choose_overall_label(
            front_status,
            back_status,
            allow_review=bool(self.args.allow_review),
        )
        if overall_final == "no_detection":
            if self.violation_event_logged:
                self.last_violation_log_message = "violation log state reset by no_detection"
            self.violation_event_logged = False
            return
        if overall_final != "violation":
            return
        if self.violation_event_logged:
            self.last_violation_log_message = "violation already logged; waiting for no_detection"
            return
        if not has_loggable_violation(front_status, back_status):
            self.last_violation_log_message = (
                "waiting for phone and lens/sticker detection before saving violation"
            )
            return

        with self.log_lock:
            front_status, front_frame, front_crop = self.frame_stores[front_index].snapshot_for_logging()
            back_status, back_frame, back_crop = self.frame_stores[back_index].snapshot_for_logging()
            if front_frame is None or back_frame is None:
                self.last_violation_log_message = "waiting for both camera frames before saving violation"
                return
            if not front_status.get("active") or not back_status.get("active"):
                self.last_violation_log_message = "waiting for both cameras to become active before saving violation"
                return
            overall_final, overall_reason = choose_overall_label(
                front_status,
                back_status,
                allow_review=bool(self.args.allow_review),
            )
            if overall_final == "no_detection":
                if self.violation_event_logged:
                    self.last_violation_log_message = "violation log state reset by no_detection"
                self.violation_event_logged = False
                return
            if overall_final != "violation":
                return
            if self.violation_event_logged:
                self.last_violation_log_message = "violation already logged; waiting for no_detection"
                return
            if not has_loggable_violation(front_status, back_status):
                self.last_violation_log_message = (
                    "waiting for phone and lens/sticker detection before saving violation"
                )
                return
            self.save_violation_event(
                front_status=front_status,
                back_status=back_status,
                front_frame=front_frame,
                back_frame=back_frame,
                front_crop=front_crop,
                back_crop=back_crop,
                overall_final=overall_final,
                overall_reason=overall_reason,
            )
            self.violation_event_logged = True

    def save_violation_event(
        self,
        *,
        front_status: dict[str, object],
        back_status: dict[str, object],
        front_frame: np.ndarray,
        back_frame: np.ndarray,
        front_crop: np.ndarray | None,
        back_crop: np.ndarray | None,
        overall_final: str,
        overall_reason: str,
    ) -> None:
        front_name = f"CAM {self.camera_indexes[0]}"
        back_name = f"CAM {self.camera_indexes[1]}"
        camera_view = build_camera_view(front_frame, back_frame, front_status, back_status, front_name, back_name)
        crop_view = build_crop_view(front_crop, back_crop, front_name, back_name, output_width=camera_view.shape[1])
        combined = build_combined_view(camera_view, crop_view, overall_final, overall_reason)

        captured_at = datetime.now()
        stamp = captured_at.strftime("%H%M%S_%f")[:-3]
        paths = make_daily_violation_paths(self.args.violation_log_dir, captured_at)
        paths["day_dir"].mkdir(parents=True, exist_ok=True)
        paths["snapshots"].mkdir(parents=True, exist_ok=True)
        paths["crops"].mkdir(parents=True, exist_ok=True)

        snapshot_path = paths["snapshots"] / f"{stamp}_violation.jpg"
        front_crop_path = paths["crops"] / f"{stamp}_front_crop.jpg"
        back_crop_path = paths["crops"] / f"{stamp}_back_crop.jpg"

        saved_snapshot = write_violation_image(snapshot_path, combined)
        saved_front_crop = write_violation_image(front_crop_path, front_crop)
        saved_back_crop = write_violation_image(back_crop_path, back_crop)

        row = {
            "timestamp": captured_at.isoformat(timespec="milliseconds"),
            "date": captured_at.strftime("%Y-%m-%d"),
            "time": captured_at.strftime("%H:%M:%S.%f")[:-3],
            "overall_final": overall_final,
            "overall_reason": overall_reason,
            "front_final": status_value(front_status, "fused_final"),
            "front_whole_pred": status_value(front_status, "whole_pred"),
            "front_whole_conf": format_conf(front_status, "whole_conf"),
            "front_crop_pred": status_value(front_status, "crop_pred"),
            "front_crop_conf": format_conf(front_status, "crop_conf"),
            "front_crop_reason": status_value(front_status, "crop_reason"),
            "front_fused_reason": status_value(front_status, "fused_reason"),
            "front_phones": status_value(front_status, "phones"),
            "front_lenses": status_value(front_status, "lenses"),
            "front_stickers": status_value(front_status, "stickers"),
            "front_void_pred": "",
            "front_void_conf": "",
            "front_void_reason": "",
            "back_final": status_value(back_status, "fused_final"),
            "back_whole_pred": status_value(back_status, "whole_pred"),
            "back_whole_conf": format_conf(back_status, "whole_conf"),
            "back_crop_pred": status_value(back_status, "crop_pred"),
            "back_crop_conf": format_conf(back_status, "crop_conf"),
            "back_crop_reason": status_value(back_status, "crop_reason"),
            "back_fused_reason": status_value(back_status, "fused_reason"),
            "back_phones": status_value(back_status, "phones"),
            "back_lenses": status_value(back_status, "lenses"),
            "back_stickers": status_value(back_status, "stickers"),
            "back_void_pred": "",
            "back_void_conf": "",
            "back_void_reason": "",
            "snapshot_path": saved_snapshot,
            "front_crop_path": saved_front_crop,
            "back_crop_path": saved_back_crop,
        }

        csv_exists = paths["csv"].exists()
        with paths["csv"].open("a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=VIOLATION_LOG_FIELDS)
            if not csv_exists:
                writer.writeheader()
            writer.writerow(row)

        self.last_violation_log_time = time.monotonic()
        self.last_violation_log_message = f"violation log saved: {paths['day_dir']}"
        print(self.last_violation_log_message, flush=True)

    def status_payload(self) -> dict[str, object]:
        camera_statuses = [self.frame_stores[index].snapshot_status() for index in self.camera_indexes]
        if len(camera_statuses) >= 2:
            overall_final, overall_reason = choose_overall_label(
                camera_statuses[0],
                camera_statuses[1],
                allow_review=bool(self.args.allow_review),
            )
        elif camera_statuses:
            overall_final = str(status_value(camera_statuses[0], "fused_final", "waiting"))
            overall_reason = str(status_value(camera_statuses[0], "fused_reason", "single_camera"))
        else:
            overall_final = "waiting"
            overall_reason = "waiting_for_camera"

        return {
            "cameras": camera_statuses,
            "camera_indexes": self.camera_indexes,
            "overall_final": overall_final,
            "overall_reason": overall_reason,
            "fusion": self.args.fusion,
            "device": self.args.device,
            "auto_log_violations": self.args.auto_log_violations,
            "violation_log_dir": str(self.args.violation_log_dir),
            "last_violation_log_message": self.last_violation_log_message,
        }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "WebcamDashboard/1.0"

    @property
    def dashboard(self) -> Dashboard:
        return self.server.dashboard  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path in {"/", "/logs"}:
            self.serve_static_file(WEB_UI_DIR / "index.html")
            return
        if path == "/api/status":
            self.serve_json(self.dashboard.status_payload())
            return
        if path == "/api/logs":
            self.serve_log_dates()
            return
        if path.startswith("/api/logs/"):
            self.serve_log_date(path)
            return
        if path.startswith("/logs/download/"):
            self.serve_log_csv_download(path)
            return
        if path.startswith("/stream/"):
            self.serve_stream(path, parsed.query)
            return
        if path.startswith("/static/"):
            requested = path.removeprefix("/static/")
            self.serve_static_file(WEB_UI_DIR / requested)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_date_dirs(self) -> list[Path]:
        root = self.dashboard.args.violation_log_dir
        if not root.exists():
            return []
        return sorted(
            [path for path in root.iterdir() if path.is_dir()],
            key=lambda path: path.name,
            reverse=True,
        )

    def log_date_dir_from_path(self, path: str, prefix: str) -> Path | None:
        date_text = path.removeprefix(prefix).strip("/")
        if not date_text or "/" in date_text or "\\" in date_text:
            return None
        date_dir = self.dashboard.args.violation_log_dir / date_text
        try:
            resolved = date_dir.resolve()
            root = self.dashboard.args.violation_log_dir.resolve()
        except OSError:
            return None
        if root not in resolved.parents and resolved != root:
            return None
        return resolved

    def serve_log_dates(self) -> None:
        dates = []
        for date_dir in self.log_date_dirs():
            csv_path = date_dir / VIOLATION_LOG_FILENAME
            count = 0
            if csv_path.exists():
                try:
                    with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
                        count = sum(1 for _ in csv.DictReader(file))
                except OSError:
                    count = 0
            dates.append({"date": date_dir.name, "path": str(date_dir), "count": count})
        self.serve_json({"dates": dates, "log_root": str(self.dashboard.args.violation_log_dir)})

    def serve_log_date(self, path: str) -> None:
        date_dir = self.log_date_dir_from_path(path, "/api/logs/")
        if date_dir is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid log date")
            return

        csv_path = date_dir / VIOLATION_LOG_FILENAME
        rows: list[dict[str, str]] = []
        if csv_path.exists():
            try:
                with csv_path.open("r", newline="", encoding="utf-8-sig") as file:
                    rows = list(csv.DictReader(file))
            except OSError as exc:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return

        display_rows = []
        for row in reversed(rows):
            display_rows.append(
                {
                    "timestamp": row.get("timestamp", ""),
                    "overall_final": row.get("overall_final", ""),
                    "front_crop_conf": row.get("front_crop_conf", ""),
                    "back_crop_conf": row.get("back_crop_conf", ""),
                    "front_phones": row.get("front_phones", ""),
                    "front_lenses": row.get("front_lenses", ""),
                    "front_stickers": row.get("front_stickers", ""),
                    "back_phones": row.get("back_phones", ""),
                    "back_lenses": row.get("back_lenses", ""),
                    "back_stickers": row.get("back_stickers", ""),
                    "snapshot_path": Path(row.get("snapshot_path", "")).name,
                }
            )

        self.serve_json(
            {
                "date": date_dir.name,
                "path": str(date_dir),
                "count": len(display_rows),
                "rows": display_rows,
                "csv_download_url": f"/logs/download/{date_dir.name}",
            }
        )

    def serve_log_csv_download(self, path: str) -> None:
        date_dir = self.log_date_dir_from_path(path, "/logs/download/")
        if date_dir is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid log date")
            return
        csv_path = date_dir / VIOLATION_LOG_FILENAME
        if not csv_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "CSV not found")
            return
        try:
            body = csv_path.read_bytes()
        except OSError as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Disposition", f'attachment; filename="{date_dir.name}_{VIOLATION_LOG_FILENAME}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_static_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            ui_root = WEB_UI_DIR.resolve()
            if ui_root not in resolved.parents and resolved != ui_root:
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return
            if not resolved.exists() or not resolved.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = resolved.read_bytes()
        except OSError as exc:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
            return

        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        if resolved.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif resolved.suffix in {".html", ".css"}:
            content_type = f"text/{resolved.suffix.removeprefix('.')}; charset=utf-8"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def serve_stream(self, path: str, query: str) -> None:
        parts = [part for part in path.split("/") if part]
        if len(parts) != 3:
            self.send_error(HTTPStatus.NOT_FOUND, "Stream not found")
            return

        _, camera_index_text, kind = parts
        if kind not in {"full", "crop"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Stream kind not found")
            return
        try:
            camera_index = int(camera_index_text)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid camera index")
            return

        store = self.dashboard.frame_stores.get(camera_index)
        if store is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Camera not found")
            return

        query_values = parse_qs(query)
        wait_timeout = float(query_values.get("timeout", ["2.0"])[0])

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        last_seq = -1
        while not self.dashboard.stop_event.is_set():
            try:
                jpeg, last_seq = store.wait_for_jpeg(kind, last_seq, timeout=wait_timeout)
                self.wfile.write(f"--{BOUNDARY}\r\n".encode("ascii"))
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception:
                break

    def log_message(self, format: str, *args: object) -> None:
        return


class DashboardHTTPServer(ThreadingHTTPServer):
    dashboard: Dashboard


def main() -> None:
    args = parse_args()
    args.whole_image_model = absolute_path(args.whole_image_model)
    args.det_model = absolute_path(args.det_model)
    args.crop_normal_violation_model = absolute_path(args.crop_normal_violation_model)
    args.violation_log_dir = absolute_path(args.violation_log_dir)
    validate_web_paths(args)
    args.violation_log_dir.mkdir(parents=True, exist_ok=True)

    dashboard = Dashboard(args)
    server = DashboardHTTPServer((args.host, args.port), DashboardRequestHandler)
    server.dashboard = dashboard

    def shutdown_handler(signum, frame) -> None:  # noqa: ANN001, ARG001
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    dashboard.start()
    url = f"http://{args.host}:{args.port}"
    print(f"webcam dashboard: {url}", flush=True)
    print(f"camera indexes: {', '.join(str(index) for index in dashboard.camera_indexes)}", flush=True)
    print("press Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.stop()
        server.server_close()


if __name__ == "__main__":
    main()
