"""Streamlit interface for still-image and video helmet detection."""

from collections import Counter
from importlib import import_module
from pathlib import Path
import tempfile

import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

from src.detection import predict_image
from src.video_processing import process_video


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]


def _apply_styles():
    st.markdown(
        """
        <style>
        :root { --ink: #111827; --muted: #6b7280; --line: #e5e7eb; --accent: #ef4444; --accent-dark: #dc2626; --teal: #0f766e; --canvas: #f8fafc; }
        html, body, [class*="css"] { font-family: 'Leelawadee UI', 'Tahoma', sans-serif; }
        .stApp { background: var(--canvas); color: var(--ink); color-scheme: light; }
        [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p,
        [data-testid="stCaptionContainer"], [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stSlider"] label, [data-testid="stCheckbox"] label { color: #374151 !important; }
        [data-testid="stCaptionContainer"] { color: #6b7280 !important; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stAppViewContainer"] > .main { background: var(--canvas); }
        .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
        .hero { padding: 2rem 2.2rem; border: 1px solid #e5e7eb; border-radius: 16px; background: #fff; box-shadow: 0 8px 24px rgba(15, 23, 42, .045); margin-bottom: 1.45rem; }
        .eyebrow { color: var(--teal); font-size: .76rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .6rem; }
        .hero h1 { color: var(--ink); font-size: clamp(1.8rem, 4vw, 2.45rem); line-height: 1.3; margin: 0 0 .55rem 0; font-weight: 700; }
        .hero p { color: #6b7280; font-size: 1.02rem; line-height: 1.75; margin: 0; max-width: 700px; }
        .section-title { color: #111827 !important; font-weight: 700; font-size: 1.25rem; margin: .15rem 0 .2rem; }
        .section-hint { color: #6b7280 !important; margin-bottom: 1rem; font-size: .92rem; }
        .upload-label { color: #374151; font-size: .9rem; font-weight: 600; margin: .35rem 0 .35rem; }
        .helper-text { color: #4b5563 !important; font-size: .84rem; line-height: 1.7; margin: .5rem 0 .35rem; }
        .upload-card, .settings-card, .result-card { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1.2rem 1.35rem; box-shadow: 0 4px 16px rgba(15, 23, 42, .035); }
        [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem 1.1rem; box-shadow: 0 4px 16px rgba(15, 23, 42, .035); }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p { color: #4b5563 !important; }
        [data-testid="stTabs"] [role="tablist"] { gap: .5rem; border-bottom: 1px solid var(--line); }
        [data-testid="stTabs"] button[role="tab"] { color: #4b5563 !important; border-radius: 9px 9px 0 0; padding: .72rem 1.1rem; }
        [data-testid="stTabs"] button[role="tab"] p { color: inherit !important; font-weight: 500; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: #dc2626 !important; border-bottom-color: #ef4444; font-weight: 600; }
        [data-testid="stTabs"] button[aria-selected="true"] p { color: #dc2626 !important; }
        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploaderDropzone"] { border: 1px dashed #cbd5e1 !important; border-radius: 10px !important; background: #fff !important; }
        [data-testid="stFileUploader"] section:hover,
        [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent) !important; background: #fffafa !important; }
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button p,
        [data-testid="stFileUploader"] button span,
        [data-testid="stFileUploader"] button svg { color: #fff !important; fill: #fff !important; stroke: #fff !important; opacity: 1 !important; }
        [data-testid="stFileUploader"] button { background: #ef4444 !important; border: 1px solid #ef4444 !important; border-radius: 8px !important; }
        [data-testid="stFileUploader"] button:hover,
        [data-testid="stFileUploader"] button:hover p,
        [data-testid="stFileUploader"] button:hover span,
        [data-testid="stFileUploader"] button:hover svg { color: #fff !important; fill: #fff !important; stroke: #fff !important; background: #dc2626 !important; }
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span { color: #374151 !important; opacity: 1 !important; }
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p { color: #374151 !important; }
        [data-testid="stSlider"] [data-testid="stThumbValue"] { color: #dc2626 !important; }
        .stButton > button, .stDownloadButton > button { color: #374151 !important; border: 1px solid #e5e7eb !important; border-radius: 10px; min-height: 2.8rem; font-weight: 600; background: #fff !important; }
        .stButton > button[kind="primary"], .stButton > button[kind="primary"] p { color: #fff !important; background: #ef4444 !important; border-color: #ef4444 !important; }
        .stButton > button[kind="primary"]:hover { background: #dc2626 !important; border-color: #dc2626 !important; }
        [data-testid="stMetric"] { background: white; border: 1px solid var(--line); border-radius: 12px; padding: .9rem 1rem; box-shadow: 0 3px 12px rgba(15, 23, 42, .03); }
        [data-testid="stImage"] img, [data-testid="stVideo"] { border-radius: 12px; }
        [data-testid="stAlert"] { border-radius: 10px; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
        .footer-note { color: var(--muted); margin-top: .4rem; font-size: .84rem; }
        #MainMenu, footer { visibility: hidden; }
        @media (max-width: 680px) { .block-container { padding: 1rem 1rem 2rem; } .hero { padding: 1.4rem; } .upload-card, .settings-card { padding: 1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _optional_preprocessing():
    """Load the image team's optional preprocessing module when available.

    Returns (preprocess_image, available_methods); both are None/empty when
    ``src/preprocessing.py`` is not importable.
    """
    try:
        module = import_module("src.preprocessing")
    except (ImportError, AttributeError):
        return None, ()
    preprocess = getattr(module, "preprocess_image", None)
    methods = tuple(
        method
        for method in getattr(module, "available_methods", list)()
        if method != "none"
    )
    if preprocess is None or not methods:
        return None, ()
    return preprocess, methods


def _preprocess_or_error(preprocess_image, method, image_rgb):
    array = np.asarray(image_rgb)
    processed = preprocess_image(array, method=method)
    if hasattr(processed, "__array__"):
        return processed
    raise TypeError("preprocess_image ต้องคืนภาพ RGB ที่แปลงเป็นอาร์เรย์ได้")


def _render_image_result(image_rgb, prediction):
    st.markdown('<div class="section-title">ผลการตรวจภาพ</div>', unsafe_allow_html=True)
    st.image(prediction["annotated_rgb"], caption="ภาพพร้อมกรอบตรวจจับจากโมเดล", use_container_width=True)
    detections = prediction.get("detections", [])
    counts = prediction.get("counts") or dict(Counter(item["class_name"] for item in detections))
    st.markdown("#### สถิติการตรวจจับ")
    st.metric("จำนวนกรอบที่ตรวจพบ", len(detections))
    if counts:
        class_columns = st.columns(max(1, len(counts)))
        for column, (class_name, count) in zip(class_columns, counts.items()):
            column.metric(class_name, count)
    if not detections:
        st.info("ไม่พบวัตถุที่โมเดลรู้จักในภาพนี้")
        return

    st.dataframe(
        [
            {
                "ประเภท": item["class_name"],
                "ความมั่นใจ": f"{float(item['confidence']):.1%}",
                "กรอบ (x1, y1, x2, y2)": ", ".join(str(v) for v in item["bbox_xyxy"]),
            }
            for item in detections
        ],
        use_container_width=True,
        hide_index=True,
    )


def _image_page(preprocess_methods, preprocess_image=None):
    st.markdown('<div class="section-title">ตรวจภาพ</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">อัปโหลดภาพเพื่อดูตำแหน่งและประเภทของหมวกนิรภัยที่ตรวจพบ</div>', unsafe_allow_html=True)
    upload_col, settings_col = st.columns([2, 1], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown('<div class="upload-label">เลือกรูปภาพ</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "เลือกรูปภาพ",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="image_upload",
            )
    with settings_col:
        with st.container(border=True):
            confidence = st.slider(
                "ค่าความมั่นใจขั้นต่ำ",
                0.0,
                1.0,
                0.25,
                0.01,
                help="คะแนนที่โมเดลต้องได้อย่างน้อยจึงจะแสดงกรอบตรวจพบ",
                key="image_confidence",
            )
            st.markdown(
                f'<div class="helper-text">แสดงผลตั้งแต่ {confidence:.0%} ขึ้นไป<br>ลดค่าเพื่อให้แสดงกรอบมากขึ้น หรือเพิ่มค่าเพื่อกรองผลที่ไม่แน่ใจ</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="helper-text">ตัวเลือกปรับภาพจะแสดงเมื่อมี src/preprocessing.py จากสมาชิก 04</div>',
        unsafe_allow_html=True,
    )
    use_preprocessing = False
    preprocess_method = None
    if preprocess_image is not None:
        use_preprocessing = st.checkbox("ปรับภาพก่อนตรวจ", value=False)
        if use_preprocessing:
            preprocess_method = st.selectbox(
                "วิธีปรับภาพ",
                preprocess_methods,
                index=0,
                help="เลือกวิธีปรับภาพก่อนส่งให้โมเดล (ตัวเลขจากชุด validation ชี้ว่าภาพเดิมยังแม่นที่สุด)",
            )

    if uploaded_files:
        st.caption(f"เลือกรูปแล้ว {len(uploaded_files)} รูป · แสดงตัวอย่างด้านล่าง")
        images = []
        preview_columns = st.columns(min(4, len(uploaded_files)))
        for index, uploaded in enumerate(uploaded_files):
            try:
                image_rgb = Image.open(uploaded).convert("RGB")
                images.append((uploaded.name, image_rgb))
                preview_columns[index % len(preview_columns)].image(
                    image_rgb, caption=uploaded.name, use_container_width=True
                )
            except (UnidentifiedImageError, OSError, ValueError) as error:
                st.error(f"เปิด {uploaded.name} ไม่ได้: {error}")

        if images and st.button("ตรวจภาพทั้งหมด", type="primary", key="run_image"):
            for index, (name, image_rgb) in enumerate(images, start=1):
                st.divider()
                st.markdown(f"### {index}. {name}")
                try:
                    input_rgb = image_rgb
                    if use_preprocessing and preprocess_image is not None:
                        input_rgb = _preprocess_or_error(
                            preprocess_image, preprocess_method, image_rgb
                        )
                    with st.spinner(f"กำลังตรวจ {name}..."):
                        prediction = predict_image(input_rgb, confidence=confidence)
                    if use_preprocessing and preprocess_image is not None:
                        st.caption(f"ผลตรวจนี้ใช้ภาพที่ปรับด้วยวิธี {preprocess_method} (src/preprocessing.py)")
                    _render_image_result(input_rgb, prediction)
                except FileNotFoundError as error:
                    st.error(f"ไม่พบไฟล์โมเดล: {error}")
                    break
                except Exception as error:
                    st.error(f"ตรวจ {name} ไม่สำเร็จ: {error}")


def _video_page():
    st.markdown('<div class="section-title">ตรวจวิดีโอ</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-hint">อัปโหลดคลิปเพื่อประมวลผลทีละเฟรมและรับวิดีโอที่แสดงกรอบผลตรวจ</div>', unsafe_allow_html=True)
    upload_col, settings_col = st.columns([2, 1], gap="large")
    with upload_col:
        with st.container(border=True):
            st.markdown('<div class="upload-label">เลือกคลิปวิดีโอ</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "เลือกคลิปวิดีโอ",
                type=VIDEO_TYPES,
                label_visibility="collapsed",
                key="video_upload",
            )
    with settings_col:
        with st.container(border=True):
            confidence = st.slider(
                "ค่าความมั่นใจขั้นต่ำ",
                0.0,
                1.0,
                0.25,
                0.01,
                help="คะแนนที่โมเดลต้องได้อย่างน้อยจึงจะแสดงกรอบตรวจพบ",
                key="video_confidence",
            )
            st.markdown(
                f'<div class="helper-text">แสดงผลตั้งแต่ {confidence:.0%} ขึ้นไป<br>ลดค่าเพื่อแสดงกรอบมากขึ้น หรือเพิ่มค่าเพื่อกรองผลที่ไม่แน่ใจ</div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="helper-text">คลิปจะถูกตรวจทีละเฟรมและบันทึกเป็น MP4 ระยะเวลาขึ้นกับความยาวคลิปและเครื่องที่ใช้</div>',
        unsafe_allow_html=True,
    )

    if uploaded is None:
        return
    if st.button("ประมวลผลวิดีโอ", type="primary", key="run_video"):
        progress = st.progress(0.0, text="เตรียมคลิป...")
        status = st.empty()
        try:
            suffix = Path(uploaded.name).suffix.lower() or ".mp4"
            with tempfile.TemporaryDirectory(prefix="helmet_detection_") as temp_dir:
                temp_root = Path(temp_dir)
                input_path = temp_root / f"input{suffix}"
                output_path = temp_root / "annotated_video.mp4"
                input_path.write_bytes(uploaded.getvalue())

                def update_progress(value):
                    progress.progress(min(max(float(value), 0.0), 1.0), text=f"กำลังตรวจวิดีโอ {value:.0%}")

                with st.spinner("กำลังตรวจทุกเฟรม..."):
                    details = process_video(
                        input_path,
                        lambda frame: predict_image(frame, confidence=confidence),
                        output_path,
                        progress_callback=update_progress,
                    )
                video_bytes = output_path.read_bytes()
                progress.progress(1.0, text="ตรวจวิดีโอเสร็จแล้ว")
                st.markdown('<div class="section-title">ผลการตรวจวิดีโอ</div>', unsafe_allow_html=True)
                st.video(video_bytes)
                st.markdown("#### สถิติการประมวลผล")
                summary_cols = st.columns(3)
                summary_cols[0].metric("จำนวนเฟรม", f"{details['frames_processed']:,}")
                summary_cols[1].metric("ความยาวคลิป", f"{details['duration_seconds']:.1f} วินาที")
                summary_cols[2].metric("ขนาดวิดีโอ", f"{details['width']}×{details['height']}")
                st.download_button(
                    "ดาวน์โหลดวิดีโอผลลัพธ์",
                    data=video_bytes,
                    file_name=f"{Path(uploaded.name).stem}_detected.mp4",
                    mime="video/mp4",
                )
                st.caption(f"ขนาดวิดีโอผลลัพธ์ {details['width']}×{details['height']} · {details['fps']:.2f} เฟรมต่อวินาที")
        except FileNotFoundError as error:
            status.empty()
            st.error(f"ไม่พบไฟล์โมเดลหรือไฟล์วิดีโอ: {error}")
        except Exception as error:
            status.empty()
            st.error(f"ประมวลผลวิดีโอไม่สำเร็จ: {error}")


def main():
    st.set_page_config(page_title="ระบบตรวจจับหมวกนิรภัย", layout="centered", initial_sidebar_state="collapsed")
    _apply_styles()
    st.markdown(
        """
        <section class="hero">
            <div class="eyebrow">SMART HELMET DETECTION</div>
            <h1>ระบบตรวจจับหมวกนิรภัย</h1>
            <p>ตรวจสอบการสวมหมวกจากภาพนิ่งหรือวิดีโอ พร้อมกรอบตำแหน่งและสรุปผลที่อ่านได้ง่าย</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    image_tab, video_tab = st.tabs(["ตรวจภาพ", "ตรวจวิดีโอ"])
    preprocess_image, preprocess_methods = _optional_preprocessing()
    with image_tab:
        _image_page(preprocess_methods, preprocess_image=preprocess_image)
    with video_tab:
        _video_page()
    st.divider()
    st.markdown('<div class="footer-note">ผลการตรวจเป็นการคาดการณ์จากโมเดล อาจคลาดเคลื่อนเมื่อภาพมืด เบลอ หรือวัตถุมีขนาดเล็ก</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
