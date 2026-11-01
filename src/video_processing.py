"""Process a video frame by frame with the project's image detector."""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


def _draw_video_results(frame_bgr, prediction):
    """Draw compact class boxes and per-frame counts for easier viewing."""
    detections = prediction.get("detections")
    if detections is None:
        annotated_rgb = np.asarray(prediction["annotated_rgb"])
        return cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    output = frame_bgr.copy()
    height, width = output.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.6, width / 1800))
    text_thickness = 1
    box_thickness = max(2, round(width / 540))
    padding = max(3, round(font_scale * 6))
    gap = padding

    def class_color(class_name):
        normalized_name = class_name.strip().lower()
        if normalized_name == "with helmet":
            return (0, 170, 0)  # green, in OpenCV's BGR order
        if normalized_name == "without helmet":
            return (0, 0, 220)  # red
        return (255, 120, 0)

    # Put a small per-frame summary at the top-left, including zero detections.
    counts = prediction.get("counts", {})
    summary_lines = [
        ("With Helmet: {}".format(counts.get("With Helmet", 0)), (0, 220, 0)),
        ("Without Helmet: {}".format(counts.get("Without Helmet", 0)), (80, 80, 255)),
    ]
    line_sizes = [
        cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        for text, _ in summary_lines
    ]
    panel_width = max(text_width for text_width, _ in line_sizes) + padding * 2
    line_height = max(text_height for _, text_height in line_sizes) + padding * 2
    panel_height = line_height * len(summary_lines) + padding
    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (25, 25, 25), -1)
    output = cv2.addWeighted(overlay, 0.78, output, 0.22, 0)

    for line_index, ((text, color), (_, text_height)) in enumerate(zip(summary_lines, line_sizes)):
        baseline_y = 8 + padding + line_index * line_height + text_height
        cv2.putText(output, text, (8 + padding, baseline_y), font, font_scale, color, text_thickness, cv2.LINE_AA)

    # Avoid placing box labels over the count panel when possible.
    occupied_labels = [(8, 8, 8 + panel_width, 8 + panel_height)]

    for detection in detections:
        class_name = str(detection["class_name"])
        confidence = float(detection["confidence"])
        x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color = class_color(class_name)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        label = f"{class_name} {confidence:.2f}"
        text_width, text_height = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
        label_width = min(text_width + padding * 2, width)
        label_height = text_height + padding * 2
        label_x = max(0, min(x1, width - label_width))

        preferred_y_positions = [
            y1 - label_height,
            y2 + gap,
            y1 - label_height * 2 - gap,
            y2 + label_height + gap * 2,
        ]
        label_y = max(0, min(preferred_y_positions[0], height - label_height))
        for candidate_y in preferred_y_positions:
            candidate_y = max(0, min(candidate_y, height - label_height))
            candidate_box = (label_x, candidate_y, label_x + label_width, candidate_y + label_height)
            overlaps = any(
                candidate_box[0] < other[2]
                and candidate_box[2] > other[0]
                and candidate_box[1] < other[3]
                and candidate_box[3] > other[1]
                for other in occupied_labels
            )
            if not overlaps:
                label_y = candidate_y
                occupied_labels.append(candidate_box)
                break

        cv2.rectangle(
            output,
            (label_x, label_y),
            (label_x + label_width, label_y + label_height),
            color,
            -1,
        )
        cv2.putText(
            output,
            label,
            (label_x + padding, label_y + padding + text_height),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return output


def process_video(
    video_path,
    predict_frame: Callable,
    output_path,
    progress_callback: Callable | None = None,
):
    """Detect objects in each video frame and save an annotated MP4.

    Args:
        video_path: Path to the input video.
        predict_frame: Image function such as ``src.detection.predict_image``.
            It receives one RGB NumPy image and returns a dictionary containing
            ``annotated_rgb``.
        output_path: Destination path. Use an ``.mp4`` extension.
        progress_callback: Optional function that accepts progress from 0.0 to
            1.0, suitable for a Streamlit progress bar.

    Returns:
        A dictionary with the output path and basic video information.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอ: {video_path}")
    if video_path.resolve() == output_path.resolve():
        raise ValueError("ไฟล์ผลลัพธ์ต้องไม่ใช่ไฟล์เดียวกับวิดีโอต้นฉบับ")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("ไฟล์ผลลัพธ์ต้องใช้นามสกุล .mp4")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("เปิดวิดีโอไม่ได้ กรุณาตรวจว่าไฟล์เป็นวิดีโอที่รองรับ")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("อ่านความเร็วหรือขนาดภาพจากวิดีโอไม่ได้")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("สร้างไฟล์ MP4 ไม่สำเร็จ ลองเลือกตำแหน่งบันทึกอื่น")

    frames_processed = 0
    completed = False

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break

            # OpenCV อ่านเฟรมเป็น BGR แต่ฟังก์ชันของคนที่ 3 รับภาพ RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            prediction = predict_frame(frame_rgb)
            # Draw small labels on the original frame so nearby labels overlap less.
            annotated_bgr = _draw_video_results(frame_bgr, prediction)
            if annotated_bgr.shape[:2] != (height, width):
                annotated_bgr = cv2.resize(annotated_bgr, (width, height))

            writer.write(annotated_bgr)
            frames_processed += 1

            if progress_callback is not None and total_frames > 0:
                progress_callback(min(frames_processed / total_frames, 1.0))

        if frames_processed == 0:
            raise ValueError("ไม่พบภาพในวิดีโอ หรืออ่านเฟรมไม่ได้")

        completed = True
    finally:
        capture.release()
        writer.release()
        if not completed and output_path.exists():
            output_path.unlink()

    duration_seconds = frames_processed / fps
    return {
        "output_path": str(output_path.resolve()),
        "frames_processed": frames_processed,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }
