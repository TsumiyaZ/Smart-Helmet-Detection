# Smart-Helmet-Detection

ระบบตรวจจับการสวมหมวกนิรภัยจากภาพนิ่งและวิดีโอ ด้วยโมเดล **YOLO11n** เชื่อมต่อหน้าเว็บด้วย **Streamlit**

## 1. เตรียมโปรเจกต์ก่อนเริ่มใช้งาน

### 1.1 ดาวน์โหลดโปรเจกต์

ถ้ายังไม่มีโค้ดในเครื่อง ให้เปิด PowerShell แล้วรัน:

```powershell
git clone https://github.com/TsumiyaZ/Smart-Helmet-Detection.git
cd Smart-Helmet-Detection
```

ถ้าเปิดโปรเจกต์นี้อยู่แล้ว ให้เปิด Terminal ที่ **โฟลเดอร์หลักของโปรเจกต์** ซึ่งเป็นตำแหน่งเดียวกับไฟล์ `requirements.txt` ก่อนทำขั้นตอนถัดไป

### 1.2 ติดตั้งโปรแกรมที่โค้ดต้องใช้

ตรวจว่าเครื่องเรียก Python ได้ แล้วติดตั้งแพ็กเกจจาก `requirements.txt`:

```powershell
python --version
python -m pip install -r requirements.txt
```

คำสั่งนี้ติดตั้ง `numpy`, `Pillow`, `ultralytics`, `opencv-python-headless`, `PyYAML` และ `streamlit` ซึ่งใช้กับโมเดล หน้าเว็บ การปรับภาพ และการประมวลผลวิดีโอ ติดตั้งครั้งเดียวต่อ Python environment

### 1.3 วางไฟล์โมเดล

วางไฟล์ `best.pt` ไว้ที่โฟลเดอร์หลักของโปรเจกต์ หรือใน `weights/best.pt` ฟังก์ชันใน `src/detection.py` จะค้นหาโมเดลจากตำแหน่งเหล่านี้ (ไฟล์โมเดลถูก push ไว้ใน git แล้ว เว้นแต่จะแทนด้วยเวอร์ชันใหม่จาก Colab)

## 2. คลาสที่โมเดลตรวจจับ

- `0`: With Helmet (สวมหมวกนิรภัย)
- `1`: Without Helmet (ไม่สวมหมวกนิรภัย)

## 3. รันโมเดล

### 3.1 ผ่านหน้าเว็บแอป (แนะนำ)

จากโฟลเดอร์หลักของโปรเจกต์ รัน:

```powershell
python -m streamlit run app.py
```

แล้วเปิด URL ที่ Streamlit แสดงในเทอร์มินัล (ปกติคือ `http://localhost:8501`)

### 3.2 ผ่านสคริปต์ Python

เรียกใช้ฟังก์ชันเดียวกับที่หน้าเว็บใช้:

```python
from PIL import Image
import numpy as np
from src.detection import predict_image

image_rgb = np.asarray(Image.open("ภาพของคุณ.jpg").convert("RGB"))
result = predict_image(image_rgb, confidence=0.25)

print(result["counts"])        # เช่น {"With Helmet": 3, "Without Helmet": 1}
print(result["detections"])    # รายการกรอบ + คะแนนความมั่นใจ
Image.fromarray(result["annotated_rgb"]).save("ผลตรวจ.jpg")
```

### ใช้งานบนหน้าเว็บ

- **ตรวจภาพ** — อัปโหลด JPG, PNG หรือ WEBP ปรับค่าความมั่นใจ (ค่าเริ่มต้น 0.25) แล้วกด **ตรวจภาพ** ได้ภาพที่วาดกรอบ รายการประเภท คะแนนความมั่นใจ และจำนวนกรอบที่พบ
- **ตรวจวิดีโอ** — อัปโหลด MP4, AVI, MOV หรือ MKV แล้วกด **ประมวลผลวิดีโอ** ระบบตรวจทีละเฟรม เมื่อเสร็จจะเล่นและให้ดาวน์โหลดผลเป็น MP4
- **ปรับภาพก่อนตรวจ (ค่าเริ่มต้น: ปิด)** — เปิด checkbox แล้วเลือกวิธี `clahe`, `median` หรือ `clahe+median` ได้ เพื่อเห็นความต่างด้วยตา (แนะนำให้ลองเปรียบเทียบกันสองรอบ)

> **ทำไมค่าเริ่มต้นจึงไม่ปรับภาพ?** จากผลการทดลองที่รันจริง (เก็บ SHA-256 ของโมเดลและรายชื่อภาพเพื่อพิสูจน์เงื่อนไขเดียวกัน) ปรากฏว่า**ภาพต้นฉบับให้คะแนนดีที่สุด** — ชุดข้อมูลที่ใช้ (Roboflow: *motorcycle helmet object detection*) ภาพส่วนใหญ่ถ่ายกลางวันแสงดีอยู่แล้ว การเพิ่มความตัดกันด้วย CLAHE จึงช่วยไม่ได้และทำให้แย่ลง รายละเอียดอยู่ใน `results/evaluation/`

## 4. การปรับภาพก่อนตรวจ (Image Preprocessing)

โค้ดอยู่ที่ `src/preprocessing.py` เรียกใช้แบบ CLI ได้ด้วย:

```powershell
python -m src.preprocessing ภาพต้นฉบับ.jpg ภาพผลลัพธ์.jpg --method clahe
```

### 4.1 Algorithm ที่ใช้จริง (2 หลัก)

| วิธี | Algorithm | ค่าที่ใช้ | ทำอะไร |
|---|---|---|---|
| `clahe` | **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — ขยายความตัดกันท้องถิ่น (local contrast) ทำบน **ช่องความสว่าง L ในสีพื้นที่ CIELAB** เพื่อไม่ทำสีเพี้ยน | `clip_limit=2.0`, `tile_grid_size=8×8` | ช่วยภาพแสงไม่สม่ำเสมอ/แสงน้อย ให้เห็นขอบและรายละเอียดหมวกชัดขึ้น |
| `median` | **Median Filter** (median blur 3×3, ไม่ใช่ Gaussian) | `median_ksize=3` (เป็นเฉพาะเลขคี่ ≥ 3) | ลด noise แบบ "salt-and-pepper" โดยไม่ทำให้ขอบวัตถุเบลอลงเท่ากรองแบบเฉลี่ย |
| `clahe+median` | **CLAHE + Median Filter ต่อกัน** (ปรับ contrast ก่อน แล้วลด noise) | ค่าเดียวกับทั้งสองวิธี | ทดสอบว่าทำสองขั้นตอนช่วยกันหรือไม่ (ผลจริง: ไม่ช่วย — แย่เท่า CLAHE เดี่ยว เพราะ CLAHE ทิ้ง texture ผิดปกติไว้ในภาพ) |

สิ่งที่**ไม่ได้ใช้** (บันทึกไว้เผื่อจะขยายงานต่อ): grayscale conversion, Gaussian Blur, threshold + edge detection (Canny) — grayscale จะทิ้งข้อมูลสีของหมวกที่โมเดลใช้แยกแยะ และ Gaussian Blur ทำขอบวัตถุเบลอ ซึ่งไม่เหมาะกับ object detection

### 4.2 สัญญาของฟังก์ชัน (สำคัญ)

```python
preprocess_image(image_rgb, method="clahe", clip_limit=2.0, tile_grid_size=(8, 8), median_ksize=3)
```

- **รับ** numpy array แบบ RGB, dtype `uint8` ขนาด `(H, W, 3)`
- **คืน** array ขนาดเท่าเดิม, dtype เดิม, **ไม่แก้ที่ภาพต้นฉบับ**

## 5. การวัดผลโมเดล (Model Evaluation)

ตัวเลขมาจาก `ultralytics.YOLO.val()` จริงทั้งหมด ไม่ได้เขียนขึ้นเอง ใช้:

- **โมเดล**: `best.pt` (YOLO11n, SHA-256 `91f829…4e`)
- **ชุดภาพ/ชุด data**: Roboflow *[motorcycle helmet object detection v1](https://universe.roboflow.com/projectt/motorcycle-helmet-object-detection/dataset/1)* (CC BY 4.0), รูปแบบ YOLOv8 export
- **ค่าเงื่อนไขที่คงที่ทุกการทดลอง**: conf = 0.25, imgsz = 640, และใช้โมเดลไฟล์เดียวกันเสมอ (ต่างกันเฉพาะการปรับภาพ)

### 5.1 ผลบนชุด test (360 ภาพ, 1,479 objects) — **ผลสุดท้าย รันครั้งเดียว**

| Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|
| 0.8924 | 0.8334 | 0.8385 | 0.4929 |

### 5.2 ปรับภาพทดสอบบนชุด validation (728 ภาพ, 3,070 objects) — ผลเลือกแบบ

| การทดลอง | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| **ภาพเดิม (เลือกใช้จริง)** | **0.8955** | 0.8253 | 0.8334 | **0.5079** |
| CLAHE | 0.8785 | 0.7984 | 0.7981 | 0.4823 |
| Median | 0.8849 | **0.8272** | **0.8361** | 0.5045 |
| CLAHE+Median | 0.8766 | 0.8019 | 0.8036 | 0.4836 |

**ข้อสรุป**: CLAHE ทำให้โมเดลแย่ลงทุกตัวชี้วัด ส่วน Median ต่างจากภาพเดิมน้อยมาก (mAP50 ดีขึ้นแค่ +0.003 แลกกับ Precision ที่ลดลง) → **เลือกภาพเดิมเป็นค่าเริ่มต้น** และรายงานผลสุดท้ายบนชุด test โดยไม่ปรับภาพ (--method ในหน้าเว็บปิดไว้)

### 5.3 เรียกใช้โปรแกรมวัดผลซ้ำ

ต้องมีชุดภาพ + ป้ายกำกับ แบบ YOLO (ไฟล์ data.yaml, โฟลเดอร์ `images/`, `labels/`):

```powershell
# เทียบก่อน/หลังปรับภาพด้วยวิธีที่เลือก (รัน YOLO.val() 2 รอบ: ภาพเดิม vs ภาพที่ปรับแล้ว)
python -m src.evaluate_preprocessing --data <path/to/data.yaml> --model best.pt --split val --method clahe --conf 0.25 --imgsz 640

# เตรียมชุดภาพที่ปรับแล้วก่อน (ยังไม่ต้องมีโมเดล)
python -m src.evaluate_preprocessing --data <path/to/data.yaml> --split val --method clahe --prepare-only

# ทำภาพเทียบก่อน/หลังแบบวางคู่กัน สำหรับใส่รายงาน
python -m src.make_comparison --input <path/to/images> --methods clahe median --limit 10
```

ผลจะถูกบันทึกที่ `results/evaluation/preprocessing_comparison.csv` (สะสมทุกรอบ) และไฟล์ Markdown สรุปของรอบล่าสุด ทุกแถวระบุ SHA-256 ของโมเดลและรายชื่อภาพ เพื่อพิสูจน์ว่าเทียบบนเงื่อนไขเดียวกัน คู่มือรูปแบบรายงานเต็มดูได้ที่ `results/README.md`

## 6. ข้อจำกัดที่พบ

- โมเดลอาจตรวจพลาดเมื่อภาพมืดมาก วัตถุมีขนาดเล็ก/อยู่ไกล หรือคนถูกบังส่วนใหญ่ — จาก confusion matrix ผล test จุดอ่อนหลักคือ **false positive บนพื้นหลัง** (กรอบที่วาดใส่พื้นหลังที่ไม่มีคนครอบคลุมหลัก)
- จำนวนกรอบที่เห็นบนหน้าเว็บเป็นการคาดการณ์ของโมเดล ไม่ใช่มาตรวัดความแม่นยำ — ความแม่นยำต้องวัดกับชุด test ที่มีป้ายกำกับเท่านั้น
- ชุดข้อมูลเทรนมีหมวกนิรภัยแบบมาตรฐานส่วนใหญ่ หมวกแบบอื่นที่หน้าตาต่างมาก (เช่น หมวกสีสว่างพิเศษ หรือครอบครึ่งศีรษะ) อาจยังครอบคลุมไม่หมด

## หมายเหตุ

ก่อนนำโมเดลไปใช้งาน ให้ตรวจว่าติดตั้ง dependencies สำเร็จและไฟล์ `best.pt` อยู่ในตำแหน่งที่ระบุไว้ คำสั่งติดตั้งต้องรันจาก Python environment เดียวกับที่ใช้เปิดโปรเจกต์
